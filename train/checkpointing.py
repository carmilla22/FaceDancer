TRAINING_STATE_VERSION = 1

_REQUIRED_STATE_FIELDS = {
    'version',
    'iteration',
    'epoch',
    'epoch_iteration',
    'iterations_per_epoch',
}


def _require_integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(
            '{} must be an integer greater than or equal to {}.'.format(
                name, minimum
            )
        )


def validate_checkpoint_interval(interval):
    _require_integer('checkpoint_interval', interval, minimum=1)
    return interval


def validate_training_state(state, checkpoint_id,
                            expected_iterations_per_epoch,
                            num_epochs=None):
    if not isinstance(state, dict):
        raise ValueError('Checkpoint state must be a JSON object.')

    if 'version' not in state:
        raise ValueError(
            'Legacy checkpoint state is not resumable because it lacks the '
            'versioned completed-iteration schema.'
        )

    if state['version'] != TRAINING_STATE_VERSION:
        raise ValueError(
            'Unsupported checkpoint state version {} (expected {}).'.format(
                state['version'], TRAINING_STATE_VERSION
            )
        )

    missing_fields = sorted(_REQUIRED_STATE_FIELDS.difference(state))
    if missing_fields:
        raise ValueError(
            'Checkpoint state is missing required fields: {}.'.format(
                ', '.join(missing_fields)
            )
        )

    for field in _REQUIRED_STATE_FIELDS:
        minimum = 1 if field == 'iterations_per_epoch' else 0
        _require_integer(field, state[field], minimum=minimum)

    _require_integer('checkpoint_id', checkpoint_id)
    _require_integer(
        'expected_iterations_per_epoch',
        expected_iterations_per_epoch,
        minimum=1
    )

    if state['iteration'] != checkpoint_id:
        raise ValueError(
            'Checkpoint ID {} does not match completed iteration {} in its '
            'state.'.format(checkpoint_id, state['iteration'])
        )

    if state['iterations_per_epoch'] != expected_iterations_per_epoch:
        raise ValueError(
            'Cannot resume with iterations_per_epoch={}; checkpoint used {}.'
            .format(
                expected_iterations_per_epoch,
                state['iterations_per_epoch']
            )
        )

    if state['epoch_iteration'] >= state['iterations_per_epoch']:
        raise ValueError(
            'Checkpoint epoch_iteration must be less than '
            'iterations_per_epoch.'
        )

    expected_iteration = (
        state['epoch'] * state['iterations_per_epoch']
        + state['epoch_iteration']
    )
    if state['iteration'] != expected_iteration:
        raise ValueError(
            'Checkpoint state is inconsistent: completed iteration {} does '
            'not match epoch {} iteration {}.'.format(
                state['iteration'], state['epoch'],
                state['epoch_iteration']
            )
        )

    if num_epochs is not None:
        _require_integer('num_epochs', num_epochs)
        if state['epoch'] > num_epochs or (
                state['epoch'] == num_epochs
                and state['epoch_iteration'] != 0):
            raise ValueError(
                'num_epochs={} ends before the checkpoint resume position.'
                .format(num_epochs)
            )

    return state


def make_training_state(iteration, epoch, epoch_iteration,
                        iterations_per_epoch):
    state = {
        'version': TRAINING_STATE_VERSION,
        'iteration': iteration,
        'epoch': epoch,
        'epoch_iteration': epoch_iteration,
        'iterations_per_epoch': iterations_per_epoch,
    }
    return validate_training_state(
        state,
        checkpoint_id=iteration,
        expected_iterations_per_epoch=iterations_per_epoch
    )


def training_state_after_step(iteration, epoch, epoch_iteration,
                              iterations_per_epoch):
    _require_integer('epoch', epoch)
    _require_integer('epoch_iteration', epoch_iteration)
    _require_integer('iterations_per_epoch', iterations_per_epoch, minimum=1)

    if epoch_iteration >= iterations_per_epoch:
        raise ValueError(
            'epoch_iteration must be less than iterations_per_epoch.'
        )

    next_epoch = epoch
    next_epoch_iteration = epoch_iteration + 1
    if next_epoch_iteration == iterations_per_epoch:
        next_epoch += 1
        next_epoch_iteration = 0

    return make_training_state(
        iteration + 1,
        next_epoch,
        next_epoch_iteration,
        iterations_per_epoch
    )


def training_epoch_iterations(state, num_epochs):
    validate_training_state(
        state,
        checkpoint_id=state.get('iteration') if isinstance(state, dict) else 0,
        expected_iterations_per_epoch=(
            state.get('iterations_per_epoch', 0)
            if isinstance(state, dict) else 0
        ),
        num_epochs=num_epochs
    )

    for epoch in range(state['epoch'], num_epochs):
        first_iteration = (
            state['epoch_iteration'] if epoch == state['epoch'] else 0
        )
        yield epoch, range(first_iteration, state['iterations_per_epoch'])


def periodic_checkpoint_id(iteration, checkpoint_interval):
    _require_integer('iteration', iteration)
    validate_checkpoint_interval(checkpoint_interval)
    if iteration > 0 and iteration % checkpoint_interval == 0:
        return str(iteration)
    return None


def final_checkpoint_id(iteration, last_saved_iteration):
    _require_integer('iteration', iteration)
    if last_saved_iteration is not None:
        _require_integer('last_saved_iteration', last_saved_iteration)
    if iteration == last_saved_iteration:
        return None
    return str(iteration)
