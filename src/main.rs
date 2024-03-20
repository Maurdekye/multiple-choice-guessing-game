#![allow(incomplete_features)]
#![feature(generic_arg_infer)]
#![feature(stmt_expr_attributes)]
#![feature(generic_const_exprs)]
use progress_observer::{reprint, Observer, Options};
use rand::{distributions::WeightedIndex, prelude::*};
use std::{array, cmp::Ordering, fmt::Display, ops::Neg, time::Duration};

#[derive(Debug)]
struct Quiz<const N: usize, const K: usize> {
    answers: [usize; N],
}

impl<const N: usize, const K: usize> Quiz<N, K> {
    pub fn new(rng: &mut ThreadRng) -> Self {
        let answers = array::from_fn(|_| rng.gen::<usize>() % K);
        Self { answers }
    }

    pub fn verify(&self, answers: &[usize; N]) -> usize {
        self.answers
            .iter()
            .zip(answers)
            .filter(|(&a, &b)| a == b)
            .count()
    }
}

trait QuizStrategy<const N: usize, const K: usize> {
    fn new() -> Self;
    fn guess(&self) -> &[usize; N];
    fn refine(&mut self, correct_answers: usize);
}

struct BruteForce<const N: usize, const K: usize> {
    current_guess: [usize; N],
    current_testing_question: usize,
    last_correct_count: Option<usize>,
}

impl<const N: usize, const K: usize> QuizStrategy<N, K> for BruteForce<N, K> {
    fn new() -> Self {
        Self {
            current_guess: [0; N],
            current_testing_question: 0,
            last_correct_count: None,
        }
    }

    fn guess(&self) -> &[usize; N] {
        &self.current_guess
    }

    fn refine(&mut self, correct_answers: usize) {
        let update_correct_count = 'inner: {
            if self.current_testing_question >= N {
                break 'inner true;
            }

            let Some(entry) = self.current_guess.get_mut(self.current_testing_question) else {
                break 'inner true;
            };

            let Some(last_correct_count) = &self.last_correct_count else {
                *entry += 1;
                break 'inner true;
            };

            let update_correct_count = match correct_answers.cmp(last_correct_count) {
                Ordering::Less => {
                    *entry -= 1;
                    self.current_testing_question += 1;
                    false
                }
                Ordering::Equal => {
                    *entry += 1;
                    break 'inner true;
                }
                Ordering::Greater => {
                    self.current_testing_question += 1;
                    true
                }
            };

            let Some(entry) = self.current_guess.get_mut(self.current_testing_question) else {
                break 'inner update_correct_count;
            };
            *entry += 1;
            update_correct_count
        };
        {
            #[cfg(debug)]
            if let Some(Ordering::Equal | Ordering::Greater) = self
                .current_guess
                .get(self.current_testing_question)
                .partial_cmp(&Some(&K))
            {
                panic!(
                    "Found no correct guesses for question {}",
                    self.current_testing_question
                );
            }
        }
        if update_correct_count {
            self.last_correct_count = Some(correct_answers);
        }
    }
}

struct Heuristic<const N: usize, const K: usize>
where
    [f32; N * K]: Sized,
{
    current_guess: [usize; N],
    last_guess: Option<([usize; N], usize)>,
    locked_in: [bool; N],

    /// choice k for question n: n * K + k
    heuristic: [f32; N * K],
    guesses_evaluated: [usize; N * K],
}

impl<const N: usize, const K: usize> Heuristic<N, K>
where
    [f32; N * K]: Sized,
{
    const N_U32: u32 = N as u32;

    pub fn randomize_some(&mut self, rng: &mut ThreadRng, portion: u32) {
        for (i, choice) in self.current_guess.iter_mut().enumerate() {
            if !self.locked_in[i] && rng.gen_ratio(portion.max(1), Self::N_U32) {
                let heuristic = &self.heuristic[i * K..(i + 1) * K];
                *choice = rng.sample(WeightedIndex::new(heuristic).unwrap());
            }
        }
    }
}

impl<const N: usize, const K: usize> Display for Heuristic<N, K>
where
    [f32; N * K]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "current_guess:  {:?}", self.current_guess)?;
        if let Some((last_guess, last_correct_count)) = &self.last_guess {
            writeln!(f, "last_guess:     {:?}", last_guess)?;
            writeln!(f, "last_correct_count: {:?}", last_correct_count)?;
        }
        writeln!(f, "heuristic:")?;
        for (i, chunk) in self.heuristic.chunks(K).enumerate() {
            writeln!(f, "{i}: {chunk:?}")?;
        }
        writeln!(f, "guesses_evaluated:")?;
        for (i, chunk) in self.guesses_evaluated.chunks(K).enumerate() {
            writeln!(f, "{i}: {chunk:?}")?;
        }

        Ok(())
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
struct PossibleAdjustment {
    incorrect_to_incorrect: usize,
    incorrect_to_correct: usize,
    correct_to_incorrect: usize,
}

struct PossibleAdjustments {
    score_change: isize,
    incorrect_to_incorrect_count: Option<usize>,
    counteractions: usize,
}

impl PossibleAdjustments {
    pub fn from(adjusted_count: isize, score_change: isize) -> Self {
        Self {
            score_change,
            incorrect_to_incorrect_count: Some((adjusted_count - score_change.abs()) as usize),
            counteractions: 0,
        }
    }
}

impl Iterator for PossibleAdjustments {
    type Item = PossibleAdjustment;

    fn next(&mut self) -> Option<Self::Item> {
        self.incorrect_to_incorrect_count
            .map(|incorrect_to_incorrect| {
                let result = PossibleAdjustment {
                    incorrect_to_incorrect,
                    incorrect_to_correct: self.score_change.max(0) as usize + self.counteractions,
                    correct_to_incorrect: self.score_change.neg().max(0) as usize
                        + self.counteractions,
                };
                self.counteractions += 1;
                self.incorrect_to_incorrect_count = incorrect_to_incorrect.checked_sub(2);
                result
            })
    }
}

#[cfg(test)]
mod possible_adjustments_tests;

impl<const N: usize, const K: usize> QuizStrategy<N, K> for Heuristic<N, K>
where
    [f32; N * K]: Sized,
{
    fn new() -> Self {
        Self {
            current_guess: [0; N],
            last_guess: None,
            locked_in: [false; N],
            heuristic: [0.5; N * K],
            guesses_evaluated: [0; N * K],
        }
    }

    fn guess(&self) -> &[usize; N] {
        &self.current_guess
    }

    fn refine(&mut self, correct_answers: usize) {
        // println!("{self}");
        let mut rng = thread_rng();

        let mut reverted_guess = false;

        'heuristic_eval: {
            let Some((last_guess, last_correct_count)) = &self.last_guess else {
                break 'heuristic_eval;
            };

            let adjusted: Vec<usize> = self
                .current_guess
                .iter()
                .zip(last_guess)
                .enumerate()
                .filter_map(|(i, (a, b))| (a != b).then_some(i))
                .collect();

            let adjusted_count = adjusted.len();
            let score_change = correct_answers as isize - *last_correct_count as isize;

            let possible_adjustments: Vec<_> =
                PossibleAdjustments::from(adjusted_count as isize, score_change).collect();

            let total_incorrect_to_correct: usize = possible_adjustments
                .iter()
                .map(|adjustment| adjustment.incorrect_to_correct)
                .sum();
            let expected_incorrect_to_correct =
                total_incorrect_to_correct as f32 / possible_adjustments.len() as f32;
            let total_correct_to_incorrect: usize = possible_adjustments
                .iter()
                .map(|adjustment| adjustment.correct_to_incorrect)
                .sum();
            let expected_correct_to_incorrect =
                total_correct_to_incorrect as f32 / possible_adjustments.len() as f32;
            let expected_value = expected_incorrect_to_correct - expected_correct_to_incorrect;
            let proportional_value = expected_value / adjusted_count as f32;
            let probability_of_good_answers = proportional_value / 2.0 + 0.5;

            for i in adjusted {
                let current_guess = self.current_guess[i];
                let last_guess = last_guess[i];
                let heuristic = &mut self.heuristic[i * K..(i + 1) * K];
                let certain_answer = if score_change == adjusted_count as isize {
                    Some(current_guess)
                } else if score_change.neg() == adjusted_count as isize {
                    Some(last_guess)
                } else {
                    None
                };
                if let Some(guess) = certain_answer {
                    self.locked_in[i] = true;
                    // these lines could be removed
                    heuristic.fill(0.0);
                    heuristic[guess] = 1.0;
                } else {
                    let guesses_evaluated = &mut self.guesses_evaluated[i * K..(i + 1) * K];
                    guesses_evaluated[current_guess] += 1;
                    heuristic[current_guess] += (probability_of_good_answers
                        - heuristic[current_guess])
                        / guesses_evaluated[current_guess] as f32;
                    guesses_evaluated[last_guess] += 1;
                    heuristic[last_guess] += ((1.0 - probability_of_good_answers)
                        - heuristic[last_guess])
                        / guesses_evaluated[last_guess] as f32;
                }
            }

            if expected_value < 0.0 {
                self.current_guess = last_guess.clone();
                reverted_guess = true;
            }
        }

        if !reverted_guess {
            self.last_guess = Some((self.current_guess.clone(), correct_answers));
        }
        self.randomize_some(&mut rng, Self::N_U32 >> 1);
        // println!("{self}");
    }
}

fn test_strategy<const N: usize, const K: usize, Strategy: QuizStrategy<N, K>>(
    rng: &mut ThreadRng,
) -> usize {
    let mut strategy = Strategy::new();
    let quiz = Quiz::<N, K>::new(rng);
    for iteration in 0.. {
        // dbg!(&iteration);
        // println!("{quiz:?}");
        let num_correct = quiz.verify(strategy.guess());
        // dbg!(&num_correct);
        if num_correct == N {
            return iteration;
        }
        strategy.refine(num_correct);
    }
    unreachable!("Loop runs indefinitely")
}

fn bench_strategy<const N: usize, const K: usize, Strategy: QuizStrategy<N, K>>() -> f64 {
    let mut rng = thread_rng();
    let mut total: u128 = 0;
    let mut trials: u64 = 0;
    for should_print in Observer::new_with(
        Duration::from_secs_f64(0.1),
        Options {
            run_for: Some(Duration::from_secs(30)),
            ..Default::default()
        },
    ) {
        total += test_strategy::<N, K, Strategy>(&mut rng) as u128;
        trials += 1;
        if should_print {
            let avg = total as f64 / trials as f64;
            reprint!("{trials: >10}: {avg: <20}");
        }
    }
    total as f64 / trials as f64
}

fn bench_strategies<const N: usize, const K: usize>()
where
    [(); N * K]: Sized,
{
    println!("heuristic:");
    bench_strategy::<N, K, Heuristic<_, _>>();
    println!();
    println!("brute force:");
    bench_strategy::<N, K, BruteForce<_, _>>();
    println!();
}

fn main() {
    bench_strategies::<10, 5>();
}
