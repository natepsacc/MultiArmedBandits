package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// this took me a second and I am not still 100p on the math here.

// We do see regret fall during the trials

// keep in memory
// linear upper confidence bound algorithm
// d -> number of features in a context vector.
// A_K -> for each arm K, dxd matrix that accumulates outer-products of past feature vecotrs.  Stores how often we've seen each direction in the context space
// B_K -> d-dimensional vector that accumulates the reward X feature
// lambda, small constant that makes the first A_K invertible, (regularlizaiton)

// Each arm will have an A_K, B_K/. Initially, each are empty except for a diagonal boost from lambda

// lets assume we are recommending from a set of N_arms arms, where each arm is a article to reccomend.
// For my mental model, lets assume the dimensions represent user interest, user age, user device type.

func CreateArm(d int, lambda float64) (mat.Matrix, mat.Vector) {
	A := mat.NewDense(d, d, nil)
	for i := 0; i < d; i++ {
		A.Set(i, i, lambda)
	}
	B := mat.NewVecDense(d, nil)
	return A, B
}

func makeArms(n_arms int, d int, lambda float64) []struct {
	A mat.Matrix
	B mat.Vector
} {

	arms := make([]struct {
		A mat.Matrix
		B mat.Vector
	}, n_arms)

	for arm := 0; arm < n_arms; arm++ {
		A, B := CreateArm(d, lambda)
		arms[arm] = struct {
			A mat.Matrix
			B mat.Vector
		}{A: A, B: B}
	}

	fmt.Println(arms)
	return arms
}

func getSyntheticContextVector(d int) mat.Vector {
	vec := mat.NewVecDense(d, nil)
	for i := 0; i < d; i++ {
		vec.SetVec(i, rand.Float64())
	}
	return vec
}

func outerProduct(a, b mat.Vector) mat.Matrix {
	if a.Len() != b.Len() {
		panic("Vector lengths do not match")
	}

	rows := a.Len()
	cols := b.Len()
	result := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, a.AtVec(i)*b.AtVec(j))
		}
	}
	return result
}

func main() {

	n_trials := 1000 // num trials
	n_arms := 50     // num arms
	d := 30          // num dimensions
	lambda := 0.1    // regularization parameter

	trueTheta := make([]*mat.VecDense, n_arms)
	for k := 0; k < n_arms; k++ {
		v := mat.NewVecDense(d, nil)
		for i := 0; i < d; i++ {
			v.SetVec(i, rand.NormFloat64())
		}
		trueTheta[k] = v
	}

	arms := makeArms(n_arms, d, lambda)

	for trial := 0; trial < n_trials; trial++ {
		// get context vector for this trial
		// for each arm, compute theta = A_inv * B
		// compute p = theta^T * x + alpha * sqrt(x^T * A_inv * x)
		// select arm with highest p
		// observe reward r
		// update A and B for selected arm

		contextVector := getSyntheticContextVector(d)

		// We need to store p_k for each arm to select the best one
		thisTrialsPK := make([]float64, n_arms)
		thisTrialsUncertainty := make([]float64, n_arms)
		thisTrialsTheta := make([]mat.VecDense, n_arms)

		for armIndex, arm := range arms {

			// Estimate the weights (theta) for the arm
			// Estimate the weight vector (the best guess of how context matters)
			// theta_hat_k = inverse(A_k) * b_k
			// theta_hat_k = invert(A_k) * b_k
			var A_inv mat.Dense
			err := A_inv.Inverse(arm.A)
			if err != nil {
				panic("Matrix inversion failed")
			}

			var theta mat.VecDense
			theta.MulVec(&A_inv, arm.B) //  inverse of accumulated outer product from past rewards by the context vectors seen on this arm
			thisTrialsTheta[armIndex] = theta

			mean := mat.Dot(&theta, contextVector) // this dot prod is inv accum outer prod of past rewards by current context vector

			// Quntify our uncertainty about that prediction
			// uncertainty = sqrt(x^T * A_inv * x)
			var temp mat.VecDense
			temp.MulVec(&A_inv, contextVector) // multiply accum outer prod inverse by current context vector

			uncertainty := math.Sqrt(mat.Dot(contextVector, &temp))
			thisTrialsUncertainty[armIndex] = uncertainty

			// Upper confidence bound for this arm
			alpha := 1.0 // exploration parameter
			p_k := mean + alpha*uncertainty
			thisTrialsPK[armIndex] = p_k

		}
		// get max p_k across arms and select that arm
		selectedArm := 0
		for armIndex, p_k := range thisTrialsPK {
			if p_k > thisTrialsPK[selectedArm] {
				selectedArm = armIndex
			}
		}

		// Simulate observing a reward for the selected arm
		// r = θ*_kᵀ x + noise
		reward := mat.Dot(trueTheta[selectedArm], contextVector) + rand.NormFloat64()*0.1

		// update the kit for that arm, A_K and B_K
		selectedArmStruct := &arms[selectedArm]

		// get outer product of context vector with itself  --  we are accumulating x*x^T and updating og arm with this product.
		A_chosen := outerProduct(contextVector, contextVector)

		var A_updated mat.Dense
		A_updated.Add(selectedArmStruct.A, A_chosen)
		selectedArmStruct.A = &A_updated

		b_chosen := mat.NewVecDense(d, nil)

		for i := 0; i < d; i++ {
			b_chosen.SetVec(i, reward*contextVector.AtVec(i))
		}

		var B_updated mat.VecDense
		B_updated.AddVec(selectedArmStruct.B, b_chosen)
		selectedArmStruct.B = &B_updated

		optimal := 0
		bestReward := math.Inf(-1)
		for k := 0; k < n_arms; k++ {
			r := mat.Dot(trueTheta[k], contextVector)
			if r > bestReward {
				bestReward = r
				optimal = k
			}
		}

		regret := bestReward - mat.Dot(trueTheta[selectedArm], contextVector)
		fmt.Printf("Trial %d: Selected Arm %d, Reward %.3f, Optimal Arm %d, Regret %.3f\n", trial, selectedArm, reward, optimal, regret)

	}

}
