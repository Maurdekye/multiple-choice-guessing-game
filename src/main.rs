#![feature(generic_arg_infer)]
#![feature(stmt_expr_attributes)]
use progress_observer::{reprint, Observer, Options};
use rand::prelude::*;
use std::{array, cmp::Ordering, time::Duration};

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

fn test_strategy<const N: usize, const K: usize, Strategy: QuizStrategy<N, K>>(
    rng: &mut ThreadRng,
) -> usize {
    let mut strategy = Strategy::new();
    let quiz = Quiz::<N, K>::new(rng);
    for iteration in 0.. {
        let num_correct = quiz.verify(strategy.guess());
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
            reprint!("{avg: <20}");
        }
    }
    total as f64 / trials as f64
}

fn main() {
    let ten_two = bench_strategy::<10, 2, BruteForce<_, _>>();
    dbg!(ten_two);
    let ten_three = bench_strategy::<10, 3, BruteForce<_, _>>();
    dbg!(ten_three);
    let ten_five = bench_strategy::<10, 5, BruteForce<_, _>>();
    dbg!(ten_five);
    let ten_ten = bench_strategy::<10, 10, BruteForce<_, _>>();
    dbg!(ten_ten);
}
