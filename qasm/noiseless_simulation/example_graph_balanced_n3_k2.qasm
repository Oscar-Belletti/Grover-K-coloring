OPENQASM 3;
include "stdgates.inc";
bit[3] _creg;
qubit[6] q91;
h q91[0];
h q91[1];
h q91[2];
x q91[5];
h q91[5];
x q91[3];
x q91[4];
cx q91[0], q91[1];
x q91[1];
cx q91[1], q91[3];
x q91[1];
cx q91[0], q91[1];
cx q91[0], q91[2];
x q91[2];
cx q91[2], q91[4];
x q91[2];
cx q91[0], q91[2];
ccx q91[3], q91[4], q91[5];
cx q91[0], q91[2];
x q91[2];
cx q91[2], q91[4];
x q91[2];
cx q91[0], q91[2];
cx q91[0], q91[1];
x q91[1];
cx q91[1], q91[3];
x q91[1];
cx q91[0], q91[1];
h q91[0];
x q91[0];
h q91[1];
x q91[1];
z q91[2];
ccx q91[0], q91[1], q91[2];
z q91[2];
x q91[0];
h q91[0];
x q91[1];
h q91[1];
_creg[0] = measure q91[0];
_creg[1] = measure q91[1];
_creg[2] = measure q91[2];