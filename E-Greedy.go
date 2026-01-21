package main

import (
	"os"
	"fmt"
	"math/rand"
	"github.com/go-echarts/go-echarts/v2/opts"
	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/types"
)

func Visualize(toVis []opts.LineData) {
	line := charts.NewLine()
	line.SetGlobalOptions(
		charts.WithInitializationOpts(opts.Initialization{
			Theme: types.ThemeInfographic,
		}),
		charts.WithTitleOpts(opts.Title{
			Title:    "Running Average Reward Over Time",
			Subtitle: "Epsilon-Greedy Strategy",
		}),
	)
	line.SetXAxis(makeRange(len(toVis))).AddSeries("Avg Reward", toVis)

	f, err := os.Create("visual.html")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	line.Render(f)
	return
}



func getRandomFloat() float64 {
	rand.Seed(int64(os.Getpid()) + rand.Int63())
	r := rand.Float64()
	return r
}



func createRandomArray(arms int) []float64 {
	rewardsArray := make([]float64, arms)
	for i := 0; i < arms; i++ {
		rewardsArray[i] = rand.Float64() * 10 // Random reward between 0 and 10
	}
	return rewardsArray
}


func getBestArm(arms []float64) int {
	bestArm := 0
	for i := 1; i < len(arms); i++ {
		if arms[i] > arms[bestArm] {
			bestArm = i
		}
	}
	return bestArm
}


func makeRange(n int) []string {
	result := make([]string, n)
	for i := 0; i < n; i++ {
		result[i] = fmt.Sprintf("%d", i)
	}
	return result
}


func selectArm(epsilon float64, arms int, meanReward []float64,) int {
	r := getRandomFloat()

	if r < epsilon {
		// Explore: select a random arm
		return rand.Intn(arms)
	} else {
		// Exploit: select the best arm
		return getBestArm(meanReward)
	}
}


func main() {

	arms := 100
	epsilon := 0.5
	trials := 10000

	runningMeanReward := 0.0

	meanReward := make([]float64, arms)
	sumReward := make([]float64, arms)
	exploredCount := make([]float64, arms)
	trueMeans := createRandomArray(arms)


	rewardsOverTime := make([]opts.LineData, 0)

	totalReward := 0.0

	for i := 0; i < trials; i++ {
		selectedArm := selectArm(epsilon, arms, meanReward)
		reward := rand.NormFloat64()*1 + trueMeans[selectedArm]
		exploredCount[selectedArm] += 1
		sumReward[selectedArm] += reward
		meanReward[selectedArm] = sumReward[selectedArm] / exploredCount[selectedArm]

		totalReward += reward

		runningMeanReward = totalReward / float64(i+1)

		rewardsOverTime = append(rewardsOverTime, opts.LineData{Value: runningMeanReward})

		if (epsilon > 0.00001){
			epsilon -= 0.00001
		}

		fmt.Printf("Trial %d: Pulled arm %d, Reward: %.2f, Running Avg Reward: %.2f\n", i+1, selectedArm, reward, runningMeanReward)
	}

	fmt.Println("Best arm is:", getBestArm(trueMeans))
	fmt.Println("Most chosen arm is:", getBestArm(exploredCount))
	fmt.Println("Total Reward after", trials, "trials is:", totalReward)

	Visualize(rewardsOverTime)

	
}