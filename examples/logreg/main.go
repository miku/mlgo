package main

import (
	"bytes"
	"fmt"
	"log"
	"os"

	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/linear"
)

// ConfusionMatrix allows to calculate accuracy and other metrics.
type ConfusionMatrix struct {
	truePositive  int
	trueNegative  int
	falsePositive int
	falseNegative int
}

func (cm ConfusionMatrix) Positive() int {
	return cm.truePositive + cm.falsePositive
}

func (cm ConfusionMatrix) Negative() int {
	return cm.trueNegative + cm.falseNegative
}

func (cm ConfusionMatrix) Accuracy() float64 {
	return (float64(cm.truePositive) + float64(cm.trueNegative)) / (float64(cm.Positive()) + float64(cm.Negative()))
}

func (cm ConfusionMatrix) Recall() float64 {
	return float64(cm.truePositive) / float64(cm.Positive())
}

func (cm ConfusionMatrix) Precision() float64 {
	return float64(cm.truePositive) / (float64(cm.truePositive) + float64(cm.falsePositive))
}

func (cm ConfusionMatrix) String() string {
	return fmt.Sprintf("\tPositives: %d\n\tNegatives: %d\n\tTrue Positives: %d\n\tTrue Negatives: %d\n\tFalse Positives: %d\n\tFalse Negatives: %d\n\n\tRecall: %.2f\n\tPrecision: %.2f\n\tAccuracy: %.2f\n",
		cm.Positive(), cm.Negative(), cm.truePositive, cm.trueNegative, cm.falsePositive, cm.falseNegative, cm.Recall(), cm.Precision(), cm.Accuracy())
}

// BestVersion keeps model and metrics grouped.
type BestVersion struct {
	ConfusionMatrix *ConfusionMatrix
	Boundary        float64
	Model           *linear.Logistic
	Iteration       int // How many iterations.
}

func (best BestVersion) Accuracy() float64 {
	if best.ConfusionMatrix == nil {
		return 0
	}
	return best.ConfusionMatrix.Accuracy()
}

func (best BestVersion) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "Maximum accuracy: %.2f\n\n", best.Accuracy())
	fmt.Fprintf(&buf, "with Model: %s\n\n", best.Model)
	fmt.Fprintf(&buf, "with Confusion Matrix:\n%s\n\n", best.ConfusionMatrix)
	fmt.Fprintf(&buf, "with Decision Boundary: %.2f\n", best.Boundary)
	fmt.Fprintf(&buf, "with Num Iterations: %d\n", best.Iteration)
	return buf.String()
}

func main() {
	fmt.Println("loading data")

	xTrain, yTrain, err := base.LoadDataFromCSV("train.csv")
	if err != nil {
		log.Fatal(err)
	}
	xTest, yTest, err := base.LoadDataFromCSV("test.csv")
	if err != nil {
		log.Fatal(err)
	}

	log.Println("grid search")
	var best BestVersion

	for iter := 100; iter < 3000; iter += 500 {
		for db := 0.05; db < 1.0; db += 0.02 {
			cm, model, err := learnModel(0.0001, 0.0, iter, db, xTrain, xTest, yTrain, yTest)
			if err != nil {
				log.Fatal(err)
			}
			if cm.Positive() == 0 || cm.Negative() == 0 {
				continue
			}
			if cm.Accuracy() > best.Accuracy() {
				best = BestVersion{ConfusionMatrix: cm, Boundary: db, Model: model, Iteration: iter}
			}
		}
	}
	fmt.Println(best)
}

// learnModel learns a logistic model over the given data with regularization.
func learnModel(learningRate float64, regularization float64, iterations int, decisionBoundary float64,
	xTrain, xTest [][]float64, yTrain, yTest []float64) (*ConfusionMatrix, *linear.Logistic, error) {
	cm := ConfusionMatrix{}

	model := linear.NewLogistic(base.BatchGA, learningRate, regularization, iterations,
		xTrain, yTrain)
	model.Output = os.Stderr // ioutil.Discard

	if err := model.Learn(); err != nil {
		return nil, nil, err
	}

	// Evaluate the Model on the Test data
	for i := range xTest {
		prediction, err := model.Predict(xTest[i])
		if err != nil {
			return nil, nil, err
		}
		y := int(yTest[i])
		positive := prediction[0] >= decisionBoundary

		if y == 1 && positive {
			cm.truePositive++
		}
		if y == 1 && !positive {
			cm.falseNegative++
		}
		if y == 0 && positive {
			cm.falsePositive++
		}
		if y == 0 && !positive {
			cm.trueNegative++
		}
	}
	return &cm, model, nil
}
