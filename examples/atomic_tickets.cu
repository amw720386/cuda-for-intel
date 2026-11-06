extern "C" __attribute__((global))
void atomic_tickets(unsigned *counter, unsigned *tickets) {
    unsigned i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x()
               + __nvvm_read_ptx_sreg_tid_x();
    tickets[i] = __atomic_fetch_add(counter, 1u, __ATOMIC_RELAXED);
}
