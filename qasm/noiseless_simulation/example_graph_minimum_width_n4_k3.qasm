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
gate cu1_129722908872320(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  u2(0, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(0, pi) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(0, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(0, pi) _gate_q_3;
}
gate cu1_129722949894256(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
}
gate cu1_129722911629760(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722911630144(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722911631104(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722911631776(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722911631536(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate cu1_129722911632112(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
}
gate cu1_129722911629472(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/16) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/16) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/16) _gate_q_1;
}
gate c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  cu1_129722911629760(pi/8) _gate_q_0, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722911630144(-pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cu1_129722911631104(pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722911631776(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722911631536(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cu1_129722911632112(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cu1_129722911629472(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
}
gate mcx_129722950212528 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722908872320(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722949894256(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722949944416(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
}
gate cu1_129722949944752(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949945088(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949945472(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949946144(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949947248(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958413440(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958412768(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958411424(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958410992(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958411184(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958411616(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958412624(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958412720(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722958409936(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722916651600(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722916651792(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722916651696(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722916652464(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722916652368(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722916653952(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722916653184(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722916653616(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722916654144(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722916654288(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722916651888(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949964704(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949964464(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949965472(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949965040(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949966768(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949967200(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949967392(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949967296(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949935936(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949936896(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949937376(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949937280(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949937808(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949935456(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949938480(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949938768(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722913232256(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722913231776(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722913232544(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722913231680(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722913233072(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722913233792(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722913233744(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722913233552(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722913234272(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722913231296(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722913231872(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949980512(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949981568(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949981952(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949982096(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949981520(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949982960(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949983440(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949983584(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949993232(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949993376(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate cu1_129722949994000(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/128) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/128) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/128) _gate_q_1;
}
gate mcu1_129722949943504(_gate_p_0) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7 {
  cu1_129722949944416(pi/64) _gate_q_6, _gate_q_7;
  cx _gate_q_6, _gate_q_5;
  cu1_129722949944752(-pi/64) _gate_q_5, _gate_q_7;
  cx _gate_q_6, _gate_q_5;
  cu1_129722949944416(pi/64) _gate_q_5, _gate_q_7;
  cx _gate_q_5, _gate_q_4;
  cu1_129722949945088(-pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_6, _gate_q_4;
  cu1_129722949944416(pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_5, _gate_q_4;
  cu1_129722949945472(-pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_6, _gate_q_4;
  cu1_129722949944416(pi/64) _gate_q_4, _gate_q_7;
  cx _gate_q_4, _gate_q_3;
  cu1_129722949946144(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129722949944416(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_5, _gate_q_3;
  cu1_129722949947248(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129722949944416(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_4, _gate_q_3;
  cu1_129722958413440(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129722949944416(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_5, _gate_q_3;
  cu1_129722958412768(-pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_6, _gate_q_3;
  cu1_129722949944416(pi/64) _gate_q_3, _gate_q_7;
  cx _gate_q_3, _gate_q_2;
  cu1_129722958411424(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949944416(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722958410992(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949944416(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_4, _gate_q_2;
  cu1_129722958411184(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949944416(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722958411616(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949944416(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_3, _gate_q_2;
  cu1_129722958412624(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949944416(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722958412720(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949944416(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_4, _gate_q_2;
  cu1_129722958409936(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949944416(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_5, _gate_q_2;
  cu1_129722916651600(-pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_6, _gate_q_2;
  cu1_129722949944416(pi/64) _gate_q_2, _gate_q_7;
  cx _gate_q_2, _gate_q_1;
  cu1_129722916651792(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722916651696(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722916652464(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722916652368(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_3, _gate_q_1;
  cu1_129722916653952(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722916653184(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722916653616(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722916654144(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_2, _gate_q_1;
  cu1_129722916654288(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722916651888(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722949964704(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722949964464(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_3, _gate_q_1;
  cu1_129722949965472(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722949965040(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_4, _gate_q_1;
  cu1_129722949966768(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_5, _gate_q_1;
  cu1_129722949967200(-pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_6, _gate_q_1;
  cu1_129722949944416(pi/64) _gate_q_1, _gate_q_7;
  cx _gate_q_1, _gate_q_0;
  cu1_129722949967392(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949967296(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722949935936(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949936896(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722949937376(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949937280(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722949937808(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949935456(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_2, _gate_q_0;
  cu1_129722949938480(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949938768(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722913232256(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722913231776(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722913232544(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722913231680(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722913233072(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722913233792(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_1, _gate_q_0;
  cu1_129722913233744(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722913233552(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722913234272(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722913231296(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722913231872(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949980512(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722949981568(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949981952(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_2, _gate_q_0;
  cu1_129722949982096(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949981520(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722949982960(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949983440(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_3, _gate_q_0;
  cu1_129722949983584(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949993232(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_4, _gate_q_0;
  cu1_129722949993376(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_5, _gate_q_0;
  cu1_129722949994000(-pi/64) _gate_q_0, _gate_q_7;
  cx _gate_q_6, _gate_q_0;
  cu1_129722949944416(pi/64) _gate_q_0, _gate_q_7;
}
gate mcx_gray _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7 {
  h _gate_q_7;
  mcu1_129722949943504(pi) _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4, _gate_q_5, _gate_q_6, _gate_q_7;
  h _gate_q_7;
}
gate cu1_129722877495088(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722877495472(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722949566816 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
}
gate mcx_129722889179584 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722877495088(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722877495472(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722949566816 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
gate cu1_129722949568640(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
}
gate cu1_129722949570128(_gate_p_0) _gate_q_0, _gate_q_1 {
  u1(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  u1(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  u1(-pi/4) _gate_q_1;
}
gate rcccx_dg_129722949581072 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(-2*pi, pi) _gate_q_3;
}
gate mcx_129722958367808 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3, _gate_q_4 {
  h _gate_q_4;
  cu1_129722949568640(pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  h _gate_q_4;
  cu1_129722949570128(-pi/2) _gate_q_3, _gate_q_4;
  h _gate_q_4;
  rcccx_dg_129722949581072 _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3;
  c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_4;
}
bit[8] _creg;
qubit[13] q77;
h q77[0];
h q77[1];
h q77[2];
h q77[3];
h q77[4];
h q77[5];
h q77[6];
h q77[7];
x q77[12];
h q77[12];
x q77[8];
x q77[9];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[6], q77[7], q77[8];
ccx q77[8], q77[9], q77[10];
ccx q77[6], q77[7], q77[8];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[4], q77[5], q77[9];
cx q77[0], q77[4];
x q77[4];
cx q77[1], q77[5];
x q77[5];
ccx q77[4], q77[5], q77[8];
x q77[5];
cx q77[1], q77[5];
x q77[4];
cx q77[0], q77[4];
mcx q77[8], q77[9], q77[10], q77[11];
cx q77[0], q77[4];
x q77[4];
cx q77[1], q77[5];
x q77[5];
ccx q77[4], q77[5], q77[8];
x q77[5];
cx q77[1], q77[5];
x q77[4];
cx q77[0], q77[4];
ccx q77[4], q77[5], q77[9];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[6], q77[7], q77[8];
ccx q77[8], q77[9], q77[10];
ccx q77[6], q77[7], q77[8];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
cx q77[2], q77[6];
x q77[6];
cx q77[3], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[9];
x q77[7];
cx q77[3], q77[7];
x q77[6];
cx q77[2], q77[6];
cx q77[0], q77[6];
x q77[6];
cx q77[1], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[8];
x q77[7];
cx q77[1], q77[7];
x q77[6];
cx q77[0], q77[6];
ccx q77[8], q77[9], q77[10];
cx q77[0], q77[6];
x q77[6];
cx q77[1], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[8];
x q77[7];
cx q77[1], q77[7];
x q77[6];
cx q77[0], q77[6];
cx q77[2], q77[6];
x q77[6];
cx q77[3], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[9];
x q77[7];
cx q77[3], q77[7];
x q77[6];
cx q77[2], q77[6];
ccx q77[2], q77[3], q77[9];
ccx q77[0], q77[1], q77[8];
mcx_129722950212528 q77[11], q77[10], q77[9], q77[8], q77[12];
ccx q77[0], q77[1], q77[8];
ccx q77[2], q77[3], q77[9];
cx q77[2], q77[6];
x q77[6];
cx q77[3], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[9];
x q77[7];
cx q77[3], q77[7];
x q77[6];
cx q77[2], q77[6];
cx q77[0], q77[6];
x q77[6];
cx q77[1], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[8];
x q77[7];
cx q77[1], q77[7];
x q77[6];
cx q77[0], q77[6];
ccx q77[8], q77[9], q77[10];
cx q77[0], q77[6];
x q77[6];
cx q77[1], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[8];
x q77[7];
cx q77[1], q77[7];
x q77[6];
cx q77[0], q77[6];
cx q77[2], q77[6];
x q77[6];
cx q77[3], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[9];
x q77[7];
cx q77[3], q77[7];
x q77[6];
cx q77[2], q77[6];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[6], q77[7], q77[8];
ccx q77[8], q77[9], q77[10];
ccx q77[6], q77[7], q77[8];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[4], q77[5], q77[9];
cx q77[0], q77[4];
x q77[4];
cx q77[1], q77[5];
x q77[5];
ccx q77[4], q77[5], q77[8];
x q77[5];
cx q77[1], q77[5];
x q77[4];
cx q77[0], q77[4];
mcx q77[8], q77[9], q77[10], q77[11];
cx q77[0], q77[4];
x q77[4];
cx q77[1], q77[5];
x q77[5];
ccx q77[4], q77[5], q77[8];
x q77[5];
cx q77[1], q77[5];
x q77[4];
cx q77[0], q77[4];
ccx q77[4], q77[5], q77[9];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[6], q77[7], q77[8];
ccx q77[8], q77[9], q77[10];
ccx q77[6], q77[7], q77[8];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
h q77[0];
x q77[0];
h q77[1];
x q77[1];
h q77[2];
x q77[2];
h q77[3];
x q77[3];
h q77[4];
x q77[4];
h q77[5];
x q77[5];
h q77[6];
x q77[6];
z q77[7];
mcx_gray q77[0], q77[1], q77[2], q77[3], q77[4], q77[5], q77[6], q77[7];
z q77[7];
x q77[0];
h q77[0];
x q77[1];
h q77[1];
x q77[2];
h q77[2];
x q77[3];
h q77[3];
x q77[4];
h q77[4];
x q77[5];
h q77[5];
x q77[6];
h q77[6];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[6], q77[7], q77[8];
ccx q77[8], q77[9], q77[10];
ccx q77[6], q77[7], q77[8];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[4], q77[5], q77[9];
cx q77[0], q77[4];
x q77[4];
cx q77[1], q77[5];
x q77[5];
ccx q77[4], q77[5], q77[8];
x q77[5];
cx q77[1], q77[5];
x q77[4];
cx q77[0], q77[4];
mcx q77[8], q77[9], q77[10], q77[11];
cx q77[0], q77[4];
x q77[4];
cx q77[1], q77[5];
x q77[5];
ccx q77[4], q77[5], q77[8];
x q77[5];
cx q77[1], q77[5];
x q77[4];
cx q77[0], q77[4];
ccx q77[4], q77[5], q77[9];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[6], q77[7], q77[8];
ccx q77[8], q77[9], q77[10];
ccx q77[6], q77[7], q77[8];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
cx q77[2], q77[6];
x q77[6];
cx q77[3], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[9];
x q77[7];
cx q77[3], q77[7];
x q77[6];
cx q77[2], q77[6];
cx q77[0], q77[6];
x q77[6];
cx q77[1], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[8];
x q77[7];
cx q77[1], q77[7];
x q77[6];
cx q77[0], q77[6];
ccx q77[8], q77[9], q77[10];
cx q77[0], q77[6];
x q77[6];
cx q77[1], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[8];
x q77[7];
cx q77[1], q77[7];
x q77[6];
cx q77[0], q77[6];
cx q77[2], q77[6];
x q77[6];
cx q77[3], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[9];
x q77[7];
cx q77[3], q77[7];
x q77[6];
cx q77[2], q77[6];
ccx q77[2], q77[3], q77[9];
ccx q77[0], q77[1], q77[8];
mcx_129722889179584 q77[11], q77[10], q77[9], q77[8], q77[12];
ccx q77[0], q77[1], q77[8];
ccx q77[2], q77[3], q77[9];
cx q77[2], q77[6];
x q77[6];
cx q77[3], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[9];
x q77[7];
cx q77[3], q77[7];
x q77[6];
cx q77[2], q77[6];
cx q77[0], q77[6];
x q77[6];
cx q77[1], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[8];
x q77[7];
cx q77[1], q77[7];
x q77[6];
cx q77[0], q77[6];
ccx q77[8], q77[9], q77[10];
cx q77[0], q77[6];
x q77[6];
cx q77[1], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[8];
x q77[7];
cx q77[1], q77[7];
x q77[6];
cx q77[0], q77[6];
cx q77[2], q77[6];
x q77[6];
cx q77[3], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[9];
x q77[7];
cx q77[3], q77[7];
x q77[6];
cx q77[2], q77[6];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[6], q77[7], q77[8];
ccx q77[8], q77[9], q77[10];
ccx q77[6], q77[7], q77[8];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[4], q77[5], q77[9];
cx q77[0], q77[4];
x q77[4];
cx q77[1], q77[5];
x q77[5];
ccx q77[4], q77[5], q77[8];
x q77[5];
cx q77[1], q77[5];
x q77[4];
cx q77[0], q77[4];
mcx q77[8], q77[9], q77[10], q77[11];
cx q77[0], q77[4];
x q77[4];
cx q77[1], q77[5];
x q77[5];
ccx q77[4], q77[5], q77[8];
x q77[5];
cx q77[1], q77[5];
x q77[4];
cx q77[0], q77[4];
ccx q77[4], q77[5], q77[9];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[6], q77[7], q77[8];
ccx q77[8], q77[9], q77[10];
ccx q77[6], q77[7], q77[8];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
h q77[0];
x q77[0];
h q77[1];
x q77[1];
h q77[2];
x q77[2];
h q77[3];
x q77[3];
h q77[4];
x q77[4];
h q77[5];
x q77[5];
h q77[6];
x q77[6];
z q77[7];
mcx_gray q77[0], q77[1], q77[2], q77[3], q77[4], q77[5], q77[6], q77[7];
z q77[7];
x q77[0];
h q77[0];
x q77[1];
h q77[1];
x q77[2];
h q77[2];
x q77[3];
h q77[3];
x q77[4];
h q77[4];
x q77[5];
h q77[5];
x q77[6];
h q77[6];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[6], q77[7], q77[8];
ccx q77[8], q77[9], q77[10];
ccx q77[6], q77[7], q77[8];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[4], q77[5], q77[9];
cx q77[0], q77[4];
x q77[4];
cx q77[1], q77[5];
x q77[5];
ccx q77[4], q77[5], q77[8];
x q77[5];
cx q77[1], q77[5];
x q77[4];
cx q77[0], q77[4];
mcx q77[8], q77[9], q77[10], q77[11];
cx q77[0], q77[4];
x q77[4];
cx q77[1], q77[5];
x q77[5];
ccx q77[4], q77[5], q77[8];
x q77[5];
cx q77[1], q77[5];
x q77[4];
cx q77[0], q77[4];
ccx q77[4], q77[5], q77[9];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[6], q77[7], q77[8];
ccx q77[8], q77[9], q77[10];
ccx q77[6], q77[7], q77[8];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
cx q77[2], q77[6];
x q77[6];
cx q77[3], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[9];
x q77[7];
cx q77[3], q77[7];
x q77[6];
cx q77[2], q77[6];
cx q77[0], q77[6];
x q77[6];
cx q77[1], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[8];
x q77[7];
cx q77[1], q77[7];
x q77[6];
cx q77[0], q77[6];
ccx q77[8], q77[9], q77[10];
cx q77[0], q77[6];
x q77[6];
cx q77[1], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[8];
x q77[7];
cx q77[1], q77[7];
x q77[6];
cx q77[0], q77[6];
cx q77[2], q77[6];
x q77[6];
cx q77[3], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[9];
x q77[7];
cx q77[3], q77[7];
x q77[6];
cx q77[2], q77[6];
ccx q77[2], q77[3], q77[9];
ccx q77[0], q77[1], q77[8];
mcx_129722958367808 q77[11], q77[10], q77[9], q77[8], q77[12];
ccx q77[0], q77[1], q77[8];
ccx q77[2], q77[3], q77[9];
cx q77[2], q77[6];
x q77[6];
cx q77[3], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[9];
x q77[7];
cx q77[3], q77[7];
x q77[6];
cx q77[2], q77[6];
cx q77[0], q77[6];
x q77[6];
cx q77[1], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[8];
x q77[7];
cx q77[1], q77[7];
x q77[6];
cx q77[0], q77[6];
ccx q77[8], q77[9], q77[10];
cx q77[0], q77[6];
x q77[6];
cx q77[1], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[8];
x q77[7];
cx q77[1], q77[7];
x q77[6];
cx q77[0], q77[6];
cx q77[2], q77[6];
x q77[6];
cx q77[3], q77[7];
x q77[7];
ccx q77[6], q77[7], q77[9];
x q77[7];
cx q77[3], q77[7];
x q77[6];
cx q77[2], q77[6];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[6], q77[7], q77[8];
ccx q77[8], q77[9], q77[10];
ccx q77[6], q77[7], q77[8];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[4], q77[5], q77[9];
cx q77[0], q77[4];
x q77[4];
cx q77[1], q77[5];
x q77[5];
ccx q77[4], q77[5], q77[8];
x q77[5];
cx q77[1], q77[5];
x q77[4];
cx q77[0], q77[4];
mcx q77[8], q77[9], q77[10], q77[11];
cx q77[0], q77[4];
x q77[4];
cx q77[1], q77[5];
x q77[5];
ccx q77[4], q77[5], q77[8];
x q77[5];
cx q77[1], q77[5];
x q77[4];
cx q77[0], q77[4];
ccx q77[4], q77[5], q77[9];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
ccx q77[6], q77[7], q77[8];
ccx q77[8], q77[9], q77[10];
ccx q77[6], q77[7], q77[8];
cx q77[0], q77[2];
x q77[2];
cx q77[1], q77[3];
x q77[3];
ccx q77[2], q77[3], q77[9];
x q77[3];
cx q77[1], q77[3];
x q77[2];
cx q77[0], q77[2];
h q77[0];
x q77[0];
h q77[1];
x q77[1];
h q77[2];
x q77[2];
h q77[3];
x q77[3];
h q77[4];
x q77[4];
h q77[5];
x q77[5];
h q77[6];
x q77[6];
z q77[7];
mcx_gray q77[0], q77[1], q77[2], q77[3], q77[4], q77[5], q77[6], q77[7];
z q77[7];
x q77[0];
h q77[0];
x q77[1];
h q77[1];
x q77[2];
h q77[2];
x q77[3];
h q77[3];
x q77[4];
h q77[4];
x q77[5];
h q77[5];
x q77[6];
h q77[6];
_creg[0] = measure q77[0];
_creg[1] = measure q77[1];
_creg[2] = measure q77[2];
_creg[3] = measure q77[3];
_creg[4] = measure q77[4];
_creg[5] = measure q77[5];
_creg[6] = measure q77[6];
_creg[7] = measure q77[7];
