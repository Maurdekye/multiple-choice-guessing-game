#![allow(incomplete_features)]
#![feature(generic_arg_infer)]
#![feature(stmt_expr_attributes)]
#![feature(generic_const_exprs)]
#![feature(array_windows)]
use plotters::{
    backend::BitMapBackend, chart::ChartBuilder, drawing::IntoDrawingArea,
    prelude::full_palette::*, series::LineSeries,
};
use progress_observer::{reprint, Observer, Options};
use rand::{distributions::WeightedIndex, prelude::*};
use std::{
    array,
    cmp::Ordering,
    fmt::Display,
    io::{stdout, Write},
    ops::Neg,
    time::Duration,
};

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
    type Options;

    fn new(options: Self::Options) -> Self;
    fn guess(&self) -> &[usize; N];
    fn refine(&mut self, correct_answers: usize);
}

struct BruteForce<const N: usize, const K: usize> {
    current_guess: [usize; N],
    current_testing_question: usize,
    last_correct_count: Option<usize>,
}

impl<const N: usize, const K: usize> QuizStrategy<N, K> for BruteForce<N, K> {
    type Options = ();

    fn new(_: Self::Options) -> Self {
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
    portion: usize,
    guesses_evaluated: [usize; N * K],
}

impl<const N: usize, const K: usize> Heuristic<N, K>
where
    [f32; N * K]: Sized,
{
    pub fn randomize_some(&mut self, rng: &mut ThreadRng, num_update: usize) {
        let random_indexes: Vec<_> = self
            .locked_in
            .iter()
            .enumerate()
            .filter_map(|(i, locked)| (!locked).then_some(i))
            .collect();
        for &i in random_indexes.choose_multiple(rng, num_update) {
            let heuristic = &self.heuristic[i * K..(i + 1) * K];
            self.current_guess[i] = rng.sample(WeightedIndex::new(heuristic).unwrap());
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

struct HeuristicOptions {
    portion: usize,
}

impl<const N: usize, const K: usize> QuizStrategy<N, K> for Heuristic<N, K>
where
    [f32; N * K]: Sized,
{
    type Options = HeuristicOptions;

    fn new(options: Self::Options) -> Self {
        Self {
            current_guess: [0; N],
            last_guess: None,
            locked_in: [false; N],
            heuristic: [0.5; N * K],
            guesses_evaluated: [0; N * K],
            portion: options.portion,
        }
    }

    fn guess(&self) -> &[usize; N] {
        &self.current_guess
    }

    fn refine(&mut self, correct_answers: usize) {
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
        self.randomize_some(&mut rng, self.portion);
    }
}

#[derive(Debug)]
struct SmartHeuristicLinearRefineParameters<const K: usize> {
    current_testing_question: usize,
    current_answer_index: usize,
    current_question_answer_order: [usize; K],
    last_correct_count: Option<usize>,
}

fn get_current_question_test_order<const N: usize, const K: usize>(
    heuristic: &Heuristic<N, K>,
    question: usize,
) -> [usize; K]
where
    [(); N * K]: Sized,
{
    let heuristic_table: [f32; K] = heuristic.heuristic[question * K..(question + 1) * K]
        .try_into()
        .unwrap();
    let mut heuristic_table: [(usize, f32); K] = array::from_fn(|i| (i, heuristic_table[i]));
    heuristic_table.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    heuristic_table.map(|(i, _)| i)
}

struct SmartHeuristic<const N: usize, const K: usize>
where
    [(); N * K]: Sized,
{
    inner_heuristic: Heuristic<N, K>,
    linear_refine: Option<SmartHeuristicLinearRefineParameters<K>>,
}

impl<const N: usize, const K: usize> QuizStrategy<N, K> for SmartHeuristic<N, K>
where
    [(); N * K]: Sized,
{
    type Options = HeuristicOptions;

    fn new(options: Self::Options) -> Self {
        Self {
            inner_heuristic: Heuristic::new(options),
            linear_refine: None,
        }
    }

    fn guess(&self) -> &[usize; N] {
        &self.inner_heuristic.current_guess
    }

    fn refine(&mut self, correct_answers: usize) {
        fn apply_refined_question<const N: usize, const K: usize>(
            heuristic: &mut Heuristic<N, K>,
            linear_refine: &SmartHeuristicLinearRefineParameters<K>,
        ) where
            [(); N * K]: Sized,
        {
            heuristic.current_guess[linear_refine.current_testing_question] =
                linear_refine.current_question_answer_order[linear_refine.current_answer_index];
        }

        let Some(linear_refine) = &mut self.linear_refine else {
            self.inner_heuristic.refine(correct_answers);
            if self.inner_heuristic.portion == 1 {
                let current_question_answer_order =
                    get_current_question_test_order(&self.inner_heuristic, 0);
                let mut linear_refine = SmartHeuristicLinearRefineParameters {
                    current_testing_question: 0,
                    current_answer_index: 0,
                    current_question_answer_order,
                    last_correct_count: None,
                };
                while self.inner_heuristic.locked_in[linear_refine.current_testing_question] {
                    linear_refine.current_testing_question += 1;
                }
                apply_refined_question(&mut self.inner_heuristic, &linear_refine);
                self.linear_refine = Some(linear_refine);
            }
            return;
        };

        // println!("{}", self.inner_heuristic);
        // println!("{linear_refine:#?}");

        if linear_refine.current_testing_question >= N {
            return;
        }

        let mut update_correct_count = true;

        'inner: {
            let Some(last_correct_count) = linear_refine.last_correct_count else {
                linear_refine.current_answer_index += 1;
                break 'inner;
            };

            match correct_answers.cmp(&last_correct_count) {
                Ordering::Less => {
                    linear_refine.current_answer_index -= 1;
                    apply_refined_question(&mut self.inner_heuristic, &linear_refine);
                    update_correct_count = false;
                }
                Ordering::Equal => {
                    linear_refine.current_answer_index += 1;
                    break 'inner;
                }
                Ordering::Greater => {}
            }

            linear_refine.current_answer_index = 0;
            loop {
                linear_refine.current_testing_question += 1;
                if linear_refine.current_testing_question >= N {
                    return;
                }
                if !self.inner_heuristic.locked_in[linear_refine.current_testing_question] {
                    break;
                }
            }
            linear_refine.current_question_answer_order = get_current_question_test_order(
                &self.inner_heuristic,
                linear_refine.current_testing_question,
            );
        }

        apply_refined_question(&mut self.inner_heuristic, &linear_refine);
        if update_correct_count {
            linear_refine.last_correct_count = Some(correct_answers);
        }
    }
}

fn test_strategy<const N: usize, const K: usize, Strategy: QuizStrategy<N, K>>(
    rng: &mut ThreadRng,
    collect: bool,
    options: Strategy::Options,
) -> (usize, Option<Vec<usize>>) {
    let mut datapoints = collect.then(|| Vec::new());
    let mut strategy = Strategy::new(options);
    let quiz = Quiz::<N, K>::new(rng);
    let mut iteration = 0;
    let mut highest = 0;
    loop {
        let num_correct = quiz.verify(strategy.guess());
        highest = highest.max(num_correct);
        if let Some(datapoints) = &mut datapoints {
            datapoints.push(highest);
        }
        if num_correct == N || iteration >= 10_000 {
            break;
        }
        strategy.refine(num_correct);
        iteration += 1;
    }

    (iteration, datapoints)
}

fn plot_series(plot_name: &str, series: Vec<Vec<usize>>) {
    let root = BitMapBackend::new(&plot_name, (1024, 1024)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut plot = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0..series.iter().map(|s| s.len()).max().unwrap(),
            0..*series.iter().flatten().max().unwrap(),
        )
        .unwrap();

    plot.configure_mesh().draw().unwrap();

    for (i, ((datapoints, color), delta_color)) in series
        .into_iter()
        .zip(
            [
                RED_900, BLUE_900, GREEN_900, YELLOW_900, ORANGE_900, BROWN_900, LIME_900,
            ]
            .into_iter()
            .cycle(),
        )
        .zip(
            [
                RED_500, BLUE_500, GREEN_500, YELLOW_500, ORANGE_500, BROWN_500, LIME_500,
            ]
            .into_iter()
            .cycle(),
        )
        .enumerate()
    {
        plot.draw_series(LineSeries::new(
            datapoints.clone().into_iter().enumerate(),
            color,
        ))
        .unwrap()
        .label(format!("{i}"));

        // let deltas: Vec<_> = datapoints.array_windows().map(|&[a, b]| b - a).collect();
        // plot.draw_series(LineSeries::new(deltas.into_iter().enumerate(), delta_color))
        //     .unwrap();

        root.present().unwrap();
    }
}

fn bench_strategy<const N: usize, const K: usize, Strategy: QuizStrategy<N, K>>(
    options: Strategy::Options,
) -> f64
where
    Strategy::Options: Clone,
{
    let mut rng = thread_rng();
    let mut total: u128 = 0;
    let mut trials: u64 = 0;
    for should_print in Observer::new_with(
        Duration::from_secs_f64(0.1),
        Options {
            run_for: Some(Duration::from_secs(10)),
            ..Default::default()
        },
    ) {
        total += test_strategy::<N, K, Strategy>(&mut rng, false, options.clone()).0 as u128;
        trials += 1;
        if should_print {
            let avg = total as f64 / trials as f64;
            reprint!("{trials: >10}: {avg: <20}");
        }
    }
    total as f64 / trials as f64
}

fn main() {
    // let (_, bf) = test_strategy::<1000, 32, BruteForce<_, _>>(&mut thread_rng(), true);
    // let (_, heur_1) = test_strategy::<1000, 32, Heuristic<_, _, 1>>(&mut thread_rng(), true);
    // let (_, heur_2) = test_strategy::<1000, 32, Heuristic<_, _, 2>>(&mut thread_rng(), true);
    // let (_, heur_4) = test_strategy::<1000, 32, Heuristic<_, _, 4>>(&mut thread_rng(), true);
    // let (_, heur_8) = test_strategy::<1000, 32, Heuristic<_, _, 8>>(&mut thread_rng(), true);
    // let (_, heur_16) = test_strategy::<1000, 32, Heuristic<_, _, 16>>(&mut thread_rng(), true);
    // let (_, heur_32) = test_strategy::<1000, 32, Heuristic<_, _, 32>>(&mut thread_rng(), true);
    // plot_series(
    //     &format!("randomize_1_2_4_8_16_32.png"),
    //     vec![
    //         bf.unwrap(),
    //         heur_1.unwrap(),
    //         heur_2.unwrap(),
    //         heur_4.unwrap(),
    //         heur_8.unwrap(),
    //         heur_16.unwrap(),
    //         heur_32.unwrap(),
    //     ],
    // );

    // let (_, bf) = test_strategy::<1000, 32, BruteForce<_, _>>(&mut thread_rng(), true);
    // let (_, heur) = test_strategy::<1000, 32, Heuristic<_, _>>(&mut thread_rng(), true);
    // let (_, smart_heur) = test_strategy::<1000, 32, SmartHeuristic<_, _>>(&mut thread_rng(), true);
    // plot_series(
    //     &format!("bf_heur_smartheur_2.png"),
    //     vec![
    //         bf.unwrap(),
    //         heur.unwrap(),
    //         smart_heur.unwrap(),
    //     ],
    // );

    let (_, bf) = test_strategy::<1000, 32, BruteForce<_, _>>(&mut thread_rng(), true, ());
    let (_, smart) = test_strategy::<1000, 32, SmartHeuristic<_, _>>(
        &mut thread_rng(),
        true,
        HeuristicOptions { portion: 32 },
    );
    plot_series("smart_test_2.png", vec![bf.unwrap(), smart.unwrap()]);
}
