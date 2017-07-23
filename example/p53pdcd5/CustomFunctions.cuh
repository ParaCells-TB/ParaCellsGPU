#pragma once

#include "CustomFunction.cuh"
#include "Parameters.cuh"
#include <stdio.h>

#define	CELL_NUM_FLOAT 10000000.0f

__device__ void initP0(Cell *cell, Environment *env)
{
	float P0 = (cell->getCellId() + 1.0f) / CELL_NUM_FLOAT;
	cell->setAttribute("P0", P0);
}

__device__ double vATM(float t)
{
	//return 1.2;
	//if (t>1060) return 0;
	double rtn;
	rtn = v0*exp(t / 12.0);
	rtn /= (1 + exp(t / 12.0))*(1 + exp((t - tc) / 24.0));
	return rtn;
}

__device__ float P(float t, float P0)
{
	float rtn;
	rtn = (P0 - 0.1f)*exp((t - 300.0f) / 100.0f);
	rtn /= (1.0f + exp((t - 300.0f) / 100.0f))*(1.0f + exp((t - tc) / 100.0f));
	rtn += 0.1f;
	return rtn;
}

//P53-MDM2 MODULE

__device__ float vp53(float Mdm2_395P_cyt, float ATM_star)
{
	float rtn;
	rtn = (1.0f - p0) + p0*pow(ATM_star, s0) / (pow(K0, s0) + pow(ATM_star, s0));
	rtn *= (1.0f - p1) + p1*pow(Mdm2_395P_cyt, s1) / (pow(K1, s1) + pow(Mdm2_395P_cyt, s1));
	rtn *= _vp53;
	return rtn;
}

__device__ float K2(float P)
{
	float rtn;
	rtn = r1*pow(a1*P, m1) / (1.0f + pow(a1*P, m1));
	rtn += 1.0f - r1;
	rtn *= _K2;
	return rtn;
}

__device__ float dp53(float Mdm2_nuc, float P)
{
	float rtn;
	rtn = p2*pow(Mdm2_nuc, s2) / (pow(K2(P), s2) + pow(Mdm2_nuc, s2));
	rtn += 1.0f - p2;
	rtn *= _dp53;
	return rtn;
}

__device__ float vMdm2(float p53)
{
	float rtn;
	rtn = p3*pow(p53, s3) / (pow(K3, s3) + pow(p53, s3));
	rtn += 1.0f - p3;
	rtn *= _vMdm2;
	return rtn;
}

__device__ float kp(float ATM_star)
{
	float rtn;
	rtn = p4*pow(ATM_star, s4) / (pow(K4, s4) + pow(ATM_star, s4));
	rtn += 1.0f - p4;
	rtn *= _kp;
	return rtn;
}

__device__ float f(float t, float P0)
{
	float rtn;
	rtn = 1.0f + r2*P(t, P0);
	return rtn;
}

__device__ float dATM(float p53)
{
	float rtn;
	rtn = p5*pow(p53, s5) / (pow(K5, s5) + pow(p53, s5));
	rtn += 1.0f - p5;
	rtn *= _dATM;
	return rtn;
}

__device__ void p53_Mdm2_module(float P0, float t, float dt, float &p53, float &Mdm2_cyt, float &Mdm2_nuc, float &Mdm2_395P_cyt, float &ATM_star)
{
	float _p53, _Mdm2_cyt, _Mdm2_nuc, _Mdm2_395P_cyt, _ATM_star;

	//first
	_p53 = vp53(Mdm2_395P_cyt, ATM_star) - dp53(Mdm2_nuc, P(t, P0))*p53;
	_Mdm2_cyt = vMdm2(p53) - kin*Mdm2_cyt + kout*Mdm2_nuc - kp(ATM_star)*Mdm2_cyt + kq*Mdm2_395P_cyt - dMdm2*Mdm2_cyt;
	_Mdm2_nuc = kin*Mdm2_cyt - kout*Mdm2_nuc - f(t, P0)*dMdm2*Mdm2_nuc;
	_Mdm2_395P_cyt = kp(ATM_star)*Mdm2_cyt - kq*Mdm2_395P_cyt - g0*dMdm2*Mdm2_395P_cyt;
	_ATM_star = vATM(t) - dATM(p53)*ATM_star;

	p53 += 0.5f*dt*_p53;
	Mdm2_cyt += 0.5f*dt*_Mdm2_cyt;
	Mdm2_nuc += 0.5f*dt*_Mdm2_nuc;
	Mdm2_395P_cyt += 0.5f*dt*_Mdm2_395P_cyt;
	ATM_star += 0.5f*dt*_ATM_star;

	//second
	_p53 = vp53(Mdm2_395P_cyt, ATM_star) - dp53(Mdm2_nuc, P(t, P0))*p53;
	_Mdm2_cyt = vMdm2(p53) - kin*Mdm2_cyt + kout*Mdm2_nuc - kp(ATM_star)*Mdm2_cyt + kq*Mdm2_395P_cyt - dMdm2*Mdm2_cyt;
	_Mdm2_nuc = kin*Mdm2_cyt - kout*Mdm2_nuc - f(t, P0)*dMdm2*Mdm2_nuc;
	_Mdm2_395P_cyt = kp(ATM_star)*Mdm2_cyt - kq*Mdm2_395P_cyt - g0*dMdm2*Mdm2_395P_cyt;
	_ATM_star = vATM(t) - dATM(p53)*ATM_star;

	p53 += dt*_p53;
	Mdm2_cyt += dt*_Mdm2_cyt;
	Mdm2_nuc += dt*_Mdm2_nuc;
	Mdm2_395P_cyt += dt*_Mdm2_395P_cyt;
	ATM_star += dt*_ATM_star;
}

//CELL FATE DECISION

__device__ float vka(float arrester)
{
	float rtn;
	rtn = p6*pow(arrester, s6) / (pow(K6, s6) + pow(arrester, s6));
	rtn += 1.0f - p6;
	rtn *= _vka;
	return rtn;
}

__device__ float vak(float arrester, float killer, float P)
{
	float rtn;
	rtn = p7*pow(arrester, s7) / (pow(K7, s7) + pow(arrester, s7));
	rtn += p8*pow(killer, s8) / (pow(K8, s8) + pow(killer, s8));
	rtn += p9*pow(P, s9) / (pow(K9, s9) + pow(P, s9));
	rtn += 1.0f - p7 - p8 - p9;
	rtn *= _vak;
	return rtn;
}

__device__ float K10(float P)
{
	float rtn;
	rtn = _K10 / (1.0f + pow(a2*P, m2));
	return rtn;
}

__device__ float vCytoC(float killer, float C3, float P)
{
	float rtn;
	rtn = p10*pow(killer, s10) / (pow(K10(P), s10) + pow(killer, s10));
	rtn += 1.0f - p10;
	rtn *= pow(C3, s11) / (pow(K11, s11) + pow(C3, s11));
	rtn *= _vCytoC;
	return rtn;
}

__device__ float vC3(float CytoC)
{
	float rtn;
	rtn = p12*pow(CytoC, s12) / (pow(K12, s12) + pow(CytoC, s12));
	rtn += 1.0f - p12;
	rtn *= _vC3;
	return rtn;
}

__device__ void cell_fate_decision(float P0, float t, float dt, float p53, float &killer, float &CytoC, float &C3, float &arrester, float &CytoCm)
{
	float _killer, _CytoC, _C3, _arrester, _CytoCm;

	//first
	_killer = vak(arrester, killer, P(t, P0))*arrester - vka(arrester)*killer;
	_CytoC = vCytoC(killer, C3, P(t, P0))*CytoCm - dCytoC*CytoC;
	_C3 = vC3(CytoC) - dC3*C3;

	killer += 0.5f*dt*_killer;
	CytoC += 0.5f*dt*_CytoC;
	C3 += 0.5f*dt*_C3;
	arrester = p53 - killer;
	CytoCm = CytoCtot - CytoC;

	//second
	_killer = vak(arrester, killer, P(t, P0))*arrester - vka(arrester)*killer;
	_CytoC = vCytoC(killer, C3, P(t, P0))*CytoCm - dCytoC*CytoC;
	_C3 = vC3(CytoC) - dC3*C3;

	killer += dt*_killer;
	CytoC += dt*_CytoC;
	C3 += dt*_C3;
	arrester = p53 - killer;
	CytoCm = CytoCtot - CytoC;
}

__device__ void work_kernel(Cell *cell, Environment *env)
{
	float P0 = cell->getAttribute("P0");

	//time
	float t;
	float dt;

	//Concentrations
	float p53, Mdm2_cyt, Mdm2_nuc, Mdm2_395P_cyt, ATM_star;
	float killer, CytoC, C3, arrester, CytoCm;

	//Init
	t = 0;
	dt = _dt;
	p53 = 0;
	Mdm2_cyt = 0;
	Mdm2_nuc = 0;
	Mdm2_395P_cyt = 0;
	ATM_star = 0;
	killer = 0;
	CytoC = 0;
	C3 = 0;
	arrester = 0;
	CytoCm = 0;

	for (int i = 1; i <= 4800; i++)
	{
		//Cell fate decision
		cell_fate_decision(P0, t, dt, p53, killer, CytoC, C3, arrester, CytoCm);
		//P53-Mdm2 module
		p53_Mdm2_module(P0, t, dt, p53, Mdm2_cyt, Mdm2_nuc, Mdm2_395P_cyt, ATM_star);
		//Time step
		t += dt;
		//Print
		if (cell->getCellId() == 5500 && i <= 1600)
		{
			printf("%.0f\t%.2f\t%.2f\t%.2f\n", t, C3, p53, ATM_star);
		}
	}

	cell->setAttribute("p53", p53);
}

__device__ CustomFunction customFunctions[2] = { initP0, work_kernel };
