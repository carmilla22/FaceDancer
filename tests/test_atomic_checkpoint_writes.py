import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

from utils.utils import save_checkpoint_internal, save_model_internal


class FakeModel:
    def __init__(self, architecture, weights, fail_weights=False):
        self.architecture = architecture
        self.weights = weights
        self.fail_weights = fail_weights

    def to_json(self):
        return self.architecture

    def save_weights(self, path):
        Path(path).write_bytes(self.weights)
        if self.fail_weights:
            raise RuntimeError('weight write failed')


class AtomicCheckpointWriteTest(unittest.TestCase):
    def test_coherent_checkpoint_appears_without_temporary_files(self):
        state = {
            'version': 1,
            'iteration': 10,
            'epoch': 1,
            'epoch_iteration': 0,
            'iterations_per_epoch': 10,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            checkpoint_id = save_checkpoint_internal(
                FakeModel('generator-json', b'generator-weights'),
                FakeModel('discriminator-json', b'discriminator-weights'),
                checkpoint_dir,
                state
            )

            self.assertEqual(checkpoint_id, '10')
            self.assertEqual(
                (checkpoint_dir / 'gen' / 'gen.json').read_text(),
                'generator-json'
            )
            self.assertEqual(
                (checkpoint_dir / 'gen' / 'gen_10.h5').read_bytes(),
                b'generator-weights'
            )
            self.assertEqual(
                (checkpoint_dir / 'dis' / 'dis.json').read_text(),
                'discriminator-json'
            )
            self.assertEqual(
                (checkpoint_dir / 'dis' / 'dis_10.h5').read_bytes(),
                b'discriminator-weights'
            )
            self.assertEqual(
                json.loads(
                    (checkpoint_dir / 'state' / '10.json').read_text()
                ),
                state
            )
            self.assertEqual(
                [path for path in checkpoint_dir.rglob('*')
                 if path.name.startswith('.')],
                []
            )

    def test_failed_weight_write_leaves_no_state_or_temporary_file(self):
        state = {
            'version': 1,
            'iteration': 10,
            'epoch': 1,
            'epoch_iteration': 0,
            'iterations_per_epoch': 10,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)

            with self.assertRaisesRegex(RuntimeError, 'weight write failed'):
                save_checkpoint_internal(
                    FakeModel('generator-json', b'generator-weights'),
                    FakeModel(
                        'discriminator-json', b'discriminator-weights',
                        fail_weights=True
                    ),
                    checkpoint_dir,
                    state
                )

            self.assertFalse(
                (checkpoint_dir / 'state' / '10.json').exists()
            )
            self.assertEqual(
                [path for path in checkpoint_dir.rglob('*')
                 if path.name.startswith('.')],
                []
            )

    def test_failed_replacement_preserves_existing_weight_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            weights_path = model_dir / 'gen_10.h5'
            weights_path.write_bytes(b'complete-old-weights')

            with self.assertRaisesRegex(RuntimeError, 'weight write failed'):
                save_model_internal(
                    FakeModel(
                        'generator-json', b'incomplete-new-weights',
                        fail_weights=True
                    ),
                    str(model_dir) + os.sep,
                    'gen',
                    10
                )

            self.assertEqual(
                weights_path.read_bytes(), b'complete-old-weights'
            )
            self.assertEqual(
                [path for path in model_dir.iterdir()
                 if path.name.startswith('.')],
                []
            )

    def test_existing_complete_checkpoint_is_immutable(self):
        state = {
            'version': 1,
            'iteration': 10,
            'epoch': 1,
            'epoch_iteration': 0,
            'iterations_per_epoch': 10,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            save_checkpoint_internal(
                FakeModel('old-generator-json', b'old-generator-weights'),
                FakeModel(
                    'old-discriminator-json', b'old-discriminator-weights'
                ),
                checkpoint_dir,
                state
            )

            with self.assertRaisesRegex(FileExistsError, 'already exists'):
                save_checkpoint_internal(
                    FakeModel(
                        'new-generator-json', b'new-generator-weights'
                    ),
                    FakeModel(
                        'new-discriminator-json',
                        b'new-discriminator-weights'
                    ),
                    checkpoint_dir,
                    state
                )

            self.assertEqual(
                (checkpoint_dir / 'gen' / 'gen_10.h5').read_bytes(),
                b'old-generator-weights'
            )
            self.assertEqual(
                (checkpoint_dir / 'dis' / 'dis_10.h5').read_bytes(),
                b'old-discriminator-weights'
            )
            self.assertEqual(
                json.loads(
                    (checkpoint_dir / 'state' / '10.json').read_text()
                ),
                state
            )

    def test_shared_architecture_cannot_invalidate_older_checkpoint(self):
        first_state = {
            'version': 1,
            'iteration': 10,
            'epoch': 1,
            'epoch_iteration': 0,
            'iterations_per_epoch': 10,
        }
        second_state = {
            **first_state,
            'iteration': 20,
            'epoch': 2,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            save_checkpoint_internal(
                FakeModel('generator-json', b'generator-10'),
                FakeModel('discriminator-json', b'discriminator-10'),
                checkpoint_dir,
                first_state
            )

            with self.assertRaisesRegex(ValueError, 'architecture'):
                save_checkpoint_internal(
                    FakeModel('changed-generator-json', b'generator-20'),
                    FakeModel('discriminator-json', b'discriminator-20'),
                    checkpoint_dir,
                    second_state
                )

            self.assertEqual(
                (checkpoint_dir / 'gen' / 'gen.json').read_text(),
                'generator-json'
            )
            self.assertEqual(
                (checkpoint_dir / 'gen' / 'gen_10.h5').read_bytes(),
                b'generator-10'
            )
            self.assertFalse(
                (checkpoint_dir / 'gen' / 'gen_20.h5').exists()
            )
            self.assertFalse(
                (checkpoint_dir / 'state' / '20.json').exists()
            )


if __name__ == '__main__':
    unittest.main()
