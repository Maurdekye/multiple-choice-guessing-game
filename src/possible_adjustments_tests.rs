use std::collections::HashSet;

use rand::{thread_rng, Rng};

use crate::{PossibleAdjustment, PossibleAdjustments};

#[test]
fn fuzzy() {
    let mut rng = thread_rng();
    for num_choices in 3..30 {
        for _ in 0..100_000 {
            let incorrect_to_incorrect = rng.gen_range(0..num_choices);
            let incorrect_to_correct = rng.gen_range(0..(num_choices - incorrect_to_incorrect));
            let correct_to_incorrect =
                num_choices - (incorrect_to_incorrect + incorrect_to_correct);
            let possibility = PossibleAdjustment {
                incorrect_to_incorrect,
                incorrect_to_correct,
                correct_to_incorrect,
            };
            let n_adjusted = (possibility.correct_to_incorrect
                + possibility.incorrect_to_correct
                + possibility.incorrect_to_incorrect) as isize;
            let score_change = possibility.incorrect_to_correct as isize
                - possibility.correct_to_incorrect as isize;
            let resultant_possibilities: Vec<_> =
                PossibleAdjustments::from(n_adjusted, score_change).collect();
            assert!(
                resultant_possibilities.contains(&possibility),
                "{possibility:?}, ({n_adjusted}, {score_change}) => {resultant_possibilities:?}"
            );
        }
    }
}

#[test]
fn three_three() {
    assert_eq!(
        HashSet::from_iter(PossibleAdjustments::from(3, 3)),
        HashSet::from([PossibleAdjustment {
            incorrect_to_incorrect: 0,
            incorrect_to_correct: 3,
            correct_to_incorrect: 0
        }])
    )
}

#[test]
fn three_two() {
    assert_eq!(
        HashSet::from_iter(PossibleAdjustments::from(3, 2)),
        HashSet::from([PossibleAdjustment {
            incorrect_to_incorrect: 1,
            incorrect_to_correct: 2,
            correct_to_incorrect: 0
        }])
    )
}

#[test]
fn three_one() {
    assert_eq!(
        HashSet::from_iter(PossibleAdjustments::from(3, 1)),
        HashSet::from([
            PossibleAdjustment {
                incorrect_to_incorrect: 2,
                incorrect_to_correct: 1,
                correct_to_incorrect: 0
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 0,
                incorrect_to_correct: 2,
                correct_to_incorrect: 1
            }
        ])
    )
}

#[test]
fn four_two() {
    assert_eq!(
        HashSet::from_iter(PossibleAdjustments::from(4, 2)),
        HashSet::from([
            PossibleAdjustment {
                incorrect_to_incorrect: 2,
                incorrect_to_correct: 2,
                correct_to_incorrect: 0
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 0,
                incorrect_to_correct: 3,
                correct_to_incorrect: 1
            }
        ])
    )
}

#[test]
fn four_zero() {
    assert_eq!(
        HashSet::from_iter(PossibleAdjustments::from(4, 0)),
        HashSet::from([
            PossibleAdjustment {
                incorrect_to_incorrect: 4,
                incorrect_to_correct: 0,
                correct_to_incorrect: 0
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 2,
                incorrect_to_correct: 1,
                correct_to_incorrect: 1
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 0,
                incorrect_to_correct: 2,
                correct_to_incorrect: 2
            }
        ])
    )
}

#[test]
fn five_zero() {
    assert_eq!(
        HashSet::from_iter(PossibleAdjustments::from(5, 0)),
        HashSet::from([
            PossibleAdjustment {
                incorrect_to_incorrect: 5,
                incorrect_to_correct: 0,
                correct_to_incorrect: 0
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 3,
                incorrect_to_correct: 1,
                correct_to_incorrect: 1
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 1,
                incorrect_to_correct: 2,
                correct_to_incorrect: 2
            }
        ])
    )
}

#[test]
fn five_one() {
    assert_eq!(
        HashSet::from_iter(PossibleAdjustments::from(5, 1)),
        HashSet::from([
            PossibleAdjustment {
                incorrect_to_incorrect: 4,
                incorrect_to_correct: 1,
                correct_to_incorrect: 0
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 2,
                incorrect_to_correct: 2,
                correct_to_incorrect: 1
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 0,
                incorrect_to_correct: 3,
                correct_to_incorrect: 2
            }
        ])
    )
}

#[test]
fn five_two() {
    assert_eq!(
        HashSet::from_iter(PossibleAdjustments::from(5, 2)),
        HashSet::from([
            PossibleAdjustment {
                incorrect_to_incorrect: 3,
                incorrect_to_correct: 2,
                correct_to_incorrect: 0
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 1,
                incorrect_to_correct: 3,
                correct_to_incorrect: 1
            },
        ])
    )
}

#[test]
fn five_three() {
    assert_eq!(
        HashSet::from_iter(PossibleAdjustments::from(5, 3)),
        HashSet::from([
            PossibleAdjustment {
                incorrect_to_incorrect: 2,
                incorrect_to_correct: 3,
                correct_to_incorrect: 0
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 0,
                incorrect_to_correct: 4,
                correct_to_incorrect: 1
            },
        ])
    )
}

#[test]
fn five_minus_one() {
    assert_eq!(
        HashSet::from_iter(PossibleAdjustments::from(5, -1)),
        HashSet::from([
            PossibleAdjustment {
                incorrect_to_incorrect: 4,
                incorrect_to_correct: 0,
                correct_to_incorrect: 1
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 2,
                incorrect_to_correct: 1,
                correct_to_incorrect: 2
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 0,
                incorrect_to_correct: 2,
                correct_to_incorrect: 3
            },
        ])
    )
}

#[test]
fn five_minus_two() {
    assert_eq!(
        HashSet::from_iter(PossibleAdjustments::from(5, -2)),
        HashSet::from([
            PossibleAdjustment {
                incorrect_to_incorrect: 3,
                incorrect_to_correct: 0,
                correct_to_incorrect: 2
            },
            PossibleAdjustment {
                incorrect_to_incorrect: 1,
                incorrect_to_correct: 1,
                correct_to_incorrect: 3
            },
        ])
    )
}
