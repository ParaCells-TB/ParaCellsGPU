#ifndef C_CUH
#define C_CUH

#include "ParaCellsObject.cuh"
#include "Cell.cuh"
#include "Environment.cuh"
#include "Constants.h"

#include <cstdio>

class C: public ParaCellsObject
{
private:
	float aa, TOR, Atg1, Atg8, aapro;
	
	float kin;
	float kout;
	float delta;
	float fu;
	float gl;
	float m1;
	float M1;
	float K1;
	float kp;
	float Kp;
	float K6;
	float M2;
	float m3;
	float M3;
	float K3;
	float m4;
	float M4;
	float K4;
	float ku;
	float K5;
	float kl;

	int step;
	float Atg8_delay[DELAY];
	bool isAlive;

	__device__ void accumulate_aaout(Cell *cell, float aaout)
	{
		float accumulator = cell->getAttribute("accumulator") + v * kout * aa - kin * aaout;
		cell->setAttribute("accumulator", accumulator);
	}

	__device__ float d_aa(float aaout)
	{
		return kin * aaout / v - kout * aa - delta * aa + fla(Atg8_delay[step % DELAY]) * aapro - fp(aa) * aa;
	}

	__device__ float d_TOR()
	{
		return fm(aa, TOR) * (1 - TOR) - gm(Atg1) * TOR;
	}

	__device__ float d_Atg1()
	{
		return fu * (1 - Atg1) - gu(TOR) * Atg1;
	}

	__device__ float d_Atg8()
	{
		return fl(Atg1) * (1 - Atg8) - gl * Atg8;
	}

	__device__ float d_aapro()
	{
		return fp(aa) - fla(Atg8_delay[step % DELAY]) * aapro;
	}

	__device__ float fla(float Atg8_tau)
	{
		return m1 + M1 * (pow(Atg8_tau, n1) / (pow(K1, n1) + pow(Atg8_tau, n1)));
	}

	__device__ float fp(float aa)
	{
		return kp + Kp * (aa / (K6 + aa));
	}

	__device__ float fm(float aa, float TOR)
	{
		return M2 * (pow(aa, n2) / (pow(K2(TOR), n2) + pow(aa, n2)));
	}

	__device__ float K2(float TOR)
	{
		return m3 + M3 * (pow(TOR, n3) / (pow(K3, n3) + pow(TOR, n3)));
	}

	__device__ float gm(float Atg1)
	{
		return m4 + M4 * (pow(Atg1, n4) / (pow(K4, n4) + pow(Atg1, n4)));
	}

	__device__ float gu(float TOR)
	{
		return ku * (pow(TOR, n5) / (pow(K5, n5) + pow(TOR, n5)));
	}

	__device__ float fl(float Atg1)
	{
		return kl * Atg1;
	}

	__device__ float rdeath()
	{
		return Rdea * (Kdea * Kdea / (Kdea * Kdea + aa * aa));
	}

	__device__ float rdivision()
	{
		return Rdiv * (aa * aa / (Kdiv * Kdiv + aa * aa));
	}

	__device__ bool isDied(Environment *env)
	{
		if (env->getUniformRandom() <= rdeath() * STEP_TIME) return true;
		else return false;
	}

	__device__ float generateParameter(float originalValue, Environment *env)
	{
		float random = env->getUniformRandom() * 0.3 - 0.15;
		return originalValue * (1.0 + random);
	}

public:
	__device__ C(Environment *env)
	{
		aa = 0;
		TOR = 0;
		Atg1 = 0;
		Atg8 = 0;
		aapro = 1;
		memset(Atg8_delay, 0, sizeof(Atg8_delay));

		kin = generateParameter(_kin, env);
		kout = generateParameter(_kout, env);
		delta = generateParameter(_delta, env);
		fu = generateParameter(_fu, env);
		gl = generateParameter(_gl, env);
		m1 = generateParameter(_m1, env);
		M1 = generateParameter(_MM1, env);
		K1 = generateParameter(_K1, env);
		kp = generateParameter(_kp, env);
		Kp = generateParameter(_Kp, env);
		K6 = generateParameter(_K6, env);
		M2 = generateParameter(_MM2, env);
		m3 = generateParameter(_m3, env);
		M3 = generateParameter(_M3, env);
		K3 = generateParameter(_K3, env);
		m4 = generateParameter(_m4, env);
		M4 = generateParameter(_M4, env);
		K4 = generateParameter(_K4, env);
		ku = generateParameter(_ku, env);
		K5 = generateParameter(_K5, env);
		kl = generateParameter(_kl, env);

		//Util
		step = 0;
		isAlive = 1;
	}

	__device__ void init(float aa, float TOR, float Atg1, float Atg8, float aapro)
	{
		this->aa = aa;
		this->TOR = TOR;
		this->Atg1 = Atg1;
		this->Atg8 = Atg8;
		this->aapro = aapro;
	}

	__device__ ParaCellsObject *proliferate(Environment *env)
	{
		C *rtn = new C(env);
		rtn->init(aa, TOR, Atg1, Atg8, aapro);

		return rtn;
	}

	__device__ void oneStep(Cell *cell, Environment *env)
	{
		step++;
		float aaout = env->getAttribute("aaout");
		accumulate_aaout(cell, aaout);
		float t_aa = aa + d_aa(aaout) * STEP_TIME;
		float t_TOR = TOR + d_TOR() * STEP_TIME;
		float t_Atg1 = Atg1 + d_Atg1() * STEP_TIME;
		float t_Atg8 = Atg8 + d_Atg8() * STEP_TIME;
		float t_aapro = aapro + d_aapro() * STEP_TIME;
		aa = t_aa;
		TOR = t_TOR;
		Atg1 = t_Atg1;
		Atg8 = t_Atg8;
		aapro = t_aapro;
		Atg8_delay[step % DELAY] = Atg8;
		if (isDied(env))
		{
			isAlive = 0;
			float value = cell->getAttribute("accumulator") + v * aa;
			cell->setAttribute("accumulator", value);
			aa = 0;
		}
		cell->setAttribute("aa", aa);
	}

	__device__ bool isDivided(Environment *env)
	{
		if (env->getUniformRandom() <= rdivision()) return true;
		else return false;
	}

	__device__ bool alive()
	{
		return isAlive;
	}
};

#endif
