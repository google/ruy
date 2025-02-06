int main(void)
{
    float arr[1024];
    float *A_ptr = arr;
    float *B_ptr = arr + 16;
    long long M = 32;
    long long N = 32;
    __asm__ __volatile__("smstart \n\t"
                        "zero { za } \n\t"
                        "whilelt pn8.s, xzr, %[M], vlx2 \n\t"
                        "pext{p0.s, p1.s}, pn8[0] \n\t"
                        "whilelt pn9.s, xzr, %[N], vlx2 \n\t"
                        "pext{p2.s, p3.s}, pn9[0] \n\t"
                        "ld1w{z0.s, z1.s}, pn8 / z, [%[A_ptr]] \n\t"
                        "ld1w{z2.s, z3.s}, pn9 / z, [%[B_ptr]] \n\t"
                        "fmopa za0.s, p2 / m, p0 / m, z2.s, z0.s \n\t"
                        "fmopa za1.s, p2 / m, p1 / m, z2.s, z1.s \n\t"
                        "fmopa za2.s, p3 / m, p0 / m, z3.s, z0.s \n\t"
                        "fmopa za3.s, p3 / m, p1 / m, z3.s, z1.s \n\t"
                        "smstop \n\t"
                         :
                         : [M] "r"(M), [N] "r"(N), [A_ptr] "r"(A_ptr), [B_ptr] "r"(B_ptr)
                         : "memory", "cc");
    return 0;
}
