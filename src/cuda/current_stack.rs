//! Per-thread current-context stack helpers (hardware independent).

/// Replace the top of the stack, as `cuCtxSetCurrent` does. A null key makes no
/// context current without exposing an older entry beneath it.
pub(super) fn replace_top(stack: &mut Vec<usize>, key: usize) {
    match stack.last_mut() {
        Some(top) => *top = key,
        None if key != 0 => stack.push(key),
        None => {}
    }
}

/// The current context key, or `None` when the thread has no current context.
pub(super) fn top(stack: &[usize]) -> Option<usize> {
    stack.last().copied().filter(|&key| key != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_current_does_not_expose_an_older_context() {
        let mut stack = vec![3, 4];
        replace_top(&mut stack, 0);
        assert_eq!(top(&stack), None);
        replace_top(&mut stack, 5);
        assert_eq!(top(&stack), Some(5));
        assert_eq!(stack, [3, 5]);
    }

    #[test]
    fn empty_stack_and_null_stay_empty() {
        let mut stack = Vec::new();
        replace_top(&mut stack, 0);
        assert!(stack.is_empty());
        assert_eq!(top(&stack), None);
        replace_top(&mut stack, 7);
        assert_eq!(top(&stack), Some(7));
    }
}
