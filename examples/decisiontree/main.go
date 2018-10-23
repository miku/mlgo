// Demonstrates decision tree classification

package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/trees"
)

func main() {

	var tree base.Classifier

	rand.Seed(0)

	// Load the data, with headers.
	iris, err := base.ParseCSVToInstances("iris_headers.csv", true)
	if err != nil {
		log.Fatal(err)
	}

	// Create a 60-40 training-test split
	trainData, testData := base.InstancesTrainTestSplit(iris, 0.60)

	//
	// First up, use ID3
	//
	tree = trees.NewID3DecisionTree(0.6)
	// (Parameter controls train-prune split.)

	// Train the ID3 tree
	err = tree.Fit(trainData)
	if err != nil {
		log.Fatal(err)
	}

	// Generate predictions
	predictions, err := tree.Predict(testData)
	if err != nil {
		log.Fatal(err)
	}

	// Evaluate
	fmt.Println("ID3 Performance (information gain)")
	cf, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		log.Fatal(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))
}
