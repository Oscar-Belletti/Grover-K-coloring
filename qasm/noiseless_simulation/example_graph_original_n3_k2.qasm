OPENQASM 3;
include "stdgates.inc";
gate mcx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  p(pi/8) _gate_q_0;
  p(pi/8) _gate_q_1;
  p(pi/8) _gate_q_2;
  p(pi/8) _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  p(-pi/8) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  cx _gate_q_1, _gate_q_2;
  p(-pi/8) _gate_q_2;
  cx _gate_q_0, _gate_q_2;
  p(pi/8) _gate_q_2;
  cx _gate_q_1, _gate_q_2;
  p(-pi/8) _gate_q_2;
  cx _gate_q_0, _gate_q_2;
  cx _gate_q_2, _gate_q_3;
  p(-pi/8) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  p(pi/8) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  p(-pi/8) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  p(pi/8) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  p(-pi/8) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  p(pi/8) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  p(-pi/8) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  h _gate_q_3;
}
bit[3] _creg;
qubit[7] q108;
h q108[0];
h q108[1];
h q108[2];
x q108[6];
h q108[6];
x q108[3];
x q108[4];
x q108[5];
cx q108[0], q108[1];
x q108[1];
cx q108[1], q108[3];
x q108[1];
cx q108[0], q108[1];
cx q108[0], q108[2];
x q108[2];
cx q108[2], q108[4];
x q108[2];
cx q108[0], q108[2];
x q108[5];
ccx q108[3], q108[4], q108[5];
cx q108[0], q108[2];
x q108[2];
cx q108[2], q108[3];
x q108[2];
cx q108[0], q108[2];
cx q108[0], q108[1];
x q108[1];
cx q108[1], q108[4];
x q108[1];
cx q108[0], q108[1];
mcx q108[3], q108[4], q108[5], q108[6];
cx q108[0], q108[1];
x q108[1];
cx q108[1], q108[4];
x q108[1];
cx q108[0], q108[1];
cx q108[0], q108[2];
x q108[2];
cx q108[2], q108[3];
x q108[2];
cx q108[0], q108[2];
x q108[5];
ccx q108[3], q108[4], q108[5];
cx q108[0], q108[2];
x q108[2];
cx q108[2], q108[4];
x q108[2];
cx q108[0], q108[2];
cx q108[0], q108[1];
x q108[1];
cx q108[1], q108[3];
x q108[1];
cx q108[0], q108[1];
h q108[0];
x q108[0];
h q108[1];
x q108[1];
z q108[2];
ccx q108[0], q108[1], q108[2];
z q108[2];
x q108[0];
h q108[0];
x q108[1];
h q108[1];
_creg[0] = measure q108[0];
_creg[1] = measure q108[1];
_creg[2] = measure q108[2];