import mlx.core as mx
import mlx.core.linalg as linalg

import matplotlib.pyplot as plt
import numpy as np # only for plotting

def inverseMatrix(matrix) -> mx.array:
    matrix_inv = linalg.inv(matrix, stream=mx.cpu)
    return matrix_inv

def matMul(matrixA, matrixB) -> mx.array:
    result = mx.matmul(matrixA, matrixB)
    return result

def CreateArm(dimensions, lmbda):
    A = lmbda * mx.eye(dimensions)
    B = mx.zeros((dimensions, 1))
    return A, B


def MakeArms(n_arms, dimensions, lmbda):
    arms = []
    for _ in range(n_arms):
        A, B = CreateArm(dimensions, lmbda)
        arms.append((A, B))
    return arms

def MakeSyntheticContextVector(dimenstions):
    context_vector = mx.random.normal(loc=0.0, scale=0.1, shape=(dimenstions, 1))
    return context_vector

def dotProduct(vectorA, vectorB):
    result = mx.sum(vectorA.flatten() * vectorB.flatten())
    return result



def main():

    n_trials = 10000
    n_arms = 10
    dimensions = 100
    lmbda = 0.1
    regret_list = []
    true_theta = mx.zeros((n_arms, dimensions, 1))

    for k in range(n_arms):
        regret_list.append(0.0)
        true_theta[k] = mx.random.normal(loc=0.0, scale=1.0, shape=(dimensions, 1)) / mx.sqrt(mx.array(dimensions, dtype=mx.float32))

    arms = MakeArms(n_arms, dimensions, lmbda)
    
    for t in range(n_trials):
        context_vector = MakeSyntheticContextVector(dimensions)
        thisTrialsPK =[]
        for i in range(n_arms):
            thisTrialsPK.append(0.0)

        for armIndex, arm in enumerate(arms):
            A, B = arm
            A_inv = inverseMatrix(A)
            theta_hat = matMul(A_inv, B)
            mean = dotProduct(theta_hat, context_vector)
            uncertainty = mx.sqrt(dotProduct(context_vector, matMul(A_inv, context_vector)))
            alpha = 1.0
            p_k = mean + alpha * uncertainty
            thisTrialsPK[armIndex] = p_k.item()
        
        selected_arm = mx.argmax(mx.array(thisTrialsPK))
        selected_arm_index = selected_arm.item()
        
        # reward
        reward = dotProduct(true_theta[selected_arm_index], context_vector).item()
        reward += mx.random.normal(loc=0, scale=0.1).item()

        # update selected arm
        A, B = arms[selected_arm_index]
        a_chosen = mx.matmul(context_vector, context_vector.T)
        A = A + a_chosen
        
        b_chosen = reward * context_vector
        B = B + b_chosen
        
        arms[selected_arm_index] = (A, B)
        
        # regret
        optimal = 0
        best_reward = float('-inf')
        for k in range(n_arms):
            r = dotProduct(true_theta[k], context_vector).item()
            if r > best_reward:
                best_reward = r
                optimal = k
        
        regret = best_reward - dotProduct(true_theta[selected_arm_index], context_vector).item()
        print(f"Trial {t}: Selected Arm {selected_arm_index}, Reward {reward:.3f}, Optimal Arm {optimal}, Regret {regret:.3f}")
        regret_list.append(regret)

    cumulative_regret = np.cumsum(regret_list)
    plt.plot(cumulative_regret)
    plt.xlabel('Trials')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret over Trials')
    plt.show()
    
if __name__ == "__main__":
    main()