import unittest

from train.checkpointing import (
    final_checkpoint_id, make_training_state, periodic_checkpoint_id,
    training_epoch_iterations, training_state_after_step,
    validate_checkpoint_interval, validate_training_state
)


class TrainingStateTest(unittest.TestCase):
    def test_fresh_run_position_advances_after_successful_step(self):
        state = make_training_state(0, 0, 0, 10)

        state = training_state_after_step(
            state['iteration'], state['epoch'], state['epoch_iteration'], 10
        )

        self.assertEqual(state, {
            'version': 1,
            'iteration': 1,
            'epoch': 0,
            'epoch_iteration': 1,
            'iterations_per_epoch': 10,
        })

    def test_mid_epoch_resume_uses_saved_inner_position(self):
        state = make_training_state(4, 0, 4, 10)

        resumed = validate_training_state(state, 4, 10, num_epochs=1)
        remaining_positions = list(training_epoch_iterations(resumed, 1))
        next_state = training_state_after_step(
            resumed['iteration'], resumed['epoch'],
            resumed['epoch_iteration'], 10
        )

        self.assertEqual(
            (resumed['epoch'], resumed['epoch_iteration']), (0, 4)
        )
        self.assertEqual(list(remaining_positions[0][1]), list(range(4, 10)))
        self.assertEqual(next_state['iteration'], 5)
        self.assertEqual(next_state['epoch_iteration'], 5)

    def test_epoch_boundary_resume_starts_next_epoch(self):
        state = make_training_state(10, 1, 0, 10)

        resumed = validate_training_state(state, 10, 10, num_epochs=2)
        remaining_positions = list(training_epoch_iterations(resumed, 2))

        self.assertEqual(
            (resumed['epoch'], resumed['epoch_iteration']), (1, 0)
        )
        self.assertEqual(remaining_positions[0][0], 1)
        self.assertEqual(list(remaining_positions[0][1]), list(range(10)))

    def test_final_checkpoint_can_resume_with_more_epochs(self):
        final_state = make_training_state(10, 1, 0, 10)

        resumed = validate_training_state(
            final_state, 10, 10, num_epochs=2
        )
        next_state = training_state_after_step(
            resumed['iteration'], resumed['epoch'],
            resumed['epoch_iteration'], 10
        )

        self.assertEqual(next_state['iteration'], 11)
        self.assertEqual(
            (next_state['epoch'], next_state['epoch_iteration']), (1, 1)
        )

    def test_rejects_legacy_state(self):
        with self.assertRaisesRegex(ValueError, 'Legacy checkpoint state'):
            validate_training_state(
                {'iteration': 9, 'epoch': 0}, 9, 10, num_epochs=1
            )

    def test_rejects_incompatible_or_inconsistent_state(self):
        state = make_training_state(10, 1, 0, 10)

        cases = (
            ({**state, 'version': 2}, 10, 10, 2, 'Unsupported'),
            (state, 9, 10, 2, 'does not match'),
            (state, 10, 20, 2, 'Cannot resume'),
            ({**state, 'iteration': 11}, 11, 10, 2, 'inconsistent'),
            (state, 10, 10, 0, 'ends before'),
        )
        for candidate, checkpoint_id, iterations, epochs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_training_state(
                        candidate, checkpoint_id, iterations,
                        num_epochs=epochs
                    )


class CheckpointCadenceTest(unittest.TestCase):
    @staticmethod
    def run_to_epoch_target(state, num_epochs, checkpoint_interval):
        checkpoint_ids = []
        for epoch, epoch_iterations in training_epoch_iterations(
                state, num_epochs):
            for epoch_iteration in epoch_iterations:
                state = training_state_after_step(
                    state['iteration'], epoch, epoch_iteration,
                    state['iterations_per_epoch']
                )
                checkpoint_id = periodic_checkpoint_id(
                    state['iteration'], checkpoint_interval
                )
                if checkpoint_id is not None:
                    checkpoint_ids.append(checkpoint_id)
        return state, checkpoint_ids

    def test_ten_plus_ten_resume_runs_exactly_twenty_steps(self):
        state = make_training_state(0, 0, 0, 10)

        state, first_ids = self.run_to_epoch_target(state, 1, 10)
        resumed_positions = list(training_epoch_iterations(state, 2))
        state, resumed_ids = self.run_to_epoch_target(state, 2, 10)

        self.assertEqual(first_ids, ['10'])
        self.assertEqual(resumed_positions[0][0], 1)
        self.assertEqual(resumed_positions[0][1].start, 0)
        self.assertEqual(resumed_ids, ['20'])
        self.assertEqual(state['iteration'], 20)
        self.assertEqual((state['epoch'], state['epoch_iteration']), (2, 0))

    def test_interval_checkpoint_ids_are_completed_iterations(self):
        checkpoint_ids = [
            periodic_checkpoint_id(iteration, 10)
            for iteration in range(1, 26)
        ]

        self.assertEqual(
            [checkpoint_id for checkpoint_id in checkpoint_ids
             if checkpoint_id is not None],
            ['10', '20']
        )

    def test_final_checkpoint_does_not_duplicate_interval_save(self):
        self.assertIsNone(final_checkpoint_id(20, 20))
        self.assertEqual(final_checkpoint_id(25, 20), '25')
        self.assertEqual(final_checkpoint_id(0, None), '0')

    def test_checkpoint_interval_must_be_positive(self):
        self.assertEqual(validate_checkpoint_interval(10000), 10000)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(interval=invalid):
                with self.assertRaises(ValueError):
                    validate_checkpoint_interval(invalid)


if __name__ == '__main__':
    unittest.main()
