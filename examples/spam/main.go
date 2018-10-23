package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"strings"

	"github.com/jbrukh/bayesian"
)

// Example groups a string with a class.
type Example struct {
	Value string
	Class string
}

// Dataset for spam.
type Dataset struct {
	Examples []Example
}

// ByClass returns examples of a given class.
func (ds *Dataset) ByClass(name string) (result []Example) {
	for _, ex := range ds.Examples {
		if ex.Class == name {
			result = append(result, ex)
		}
	}
	return
}

// TrainTestSplit splits the dataset.
func (ds *Dataset) TrainTestSplit(trainingPct float64) (train *Dataset, test *Dataset) {
	train, test = new(Dataset), new(Dataset)
	for _, ex := range ds.Examples {
		if rand.Float64() < trainingPct {
			train.Examples = append(train.Examples, ex)
		} else {
			test.Examples = append(test.Examples, ex)
		}
	}
	return train, test
}

func main() {
	flag.Parse()

	var example []string
	for _, arg := range flag.Args() {
		example = append(example, strings.ToLower(arg))
	}

	rand.Seed(0)
	ds := new(Dataset)
	r := csv.NewReader(os.Stdin)

	for {
		// Iterate over examples.
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		if len(record) < 2 {
			log.Fatal("to few columns: %v", record)
		}

		// Save complete examples, tokenize for model later.
		class, text := strings.TrimSpace(record[0]), strings.TrimSpace(record[1])
		ds.Examples = append(ds.Examples, Example{Class: class, Value: text})
	}

	// Train, test split.
	train, test := ds.TrainTestSplit(0.8)
	log.Printf("samples train=%d, test=%d", len(train.Examples), len(test.Examples))

	// Return the tokens for a single example, to use for train and test inputs.
	tokenize := func(example Example) (result []string) {
		for _, token := range strings.Fields(example.Value) {
			token := strings.ToLower(token)
			result = append(result, token)
		}
		return
	}

	// Tokenizes and lowercases examples and returns a slice of strings.
	// Separated out, so we can adjust the processing a bit.
	tokenizeExamples := func(examples []Example) (result []string) {
		for _, ex := range examples {
			for _, token := range tokenize(ex) {
				result = append(result, token)
			}
		}
		return
	}

	// Class 0 will be ham, 1 spam.
	classifier := bayesian.NewClassifier("ham", "spam")

	ham := tokenizeExamples(train.ByClass("ham"))
	spam := tokenizeExamples(train.ByClass("spam"))

	classifier.Learn(ham, "ham")
	classifier.Learn(spam, "spam")

	// XXX: Evaluate on test set.
	tp, tn, fp, fn := 0, 0, 0, 0
	for _, ex := range test.Examples {
		_, likely, _ := classifier.LogScores(tokenize(ex))
		switch {
		case likely == 0 && ex.Class == "ham":
			tp++
		case likely == 0 && ex.Class == "spam":
			fp++
		case likely == 1 && ex.Class == "spam":
			tn++
		case likely == 1 && ex.Class == "ham":
			fn++
		}
	}

	// Accuracy = TP + TN/ TP + FP + FN + TN
	testAccuracy := float64(tp+tn) / float64(tp+tn+fp+fn)
	log.Printf("accuracy=%0.2f", testAccuracy)

	scores, likely, _ := classifier.LogScores(example)
	log.Println(scores, likely)

	output := strings.Join(example, " ")

	switch likely {
	case 0:
		fmt.Printf("ham: %s\n", output)
	case 1:
		fmt.Printf("spam: %s\n", output)
	}
}
