// Demonstrates decision tree classification

package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/filters"
)

func main() {

	// var tree base.Classifier

	rand.Seed(0)

	// Load the data, with headers.
	iris, err := base.ParseCSVToInstances("iris_headers.csv", true)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(iris)

	// Discretise the iris dataset with Chi-Merge
	filt := filters.NewChiMergeFilter(iris, 0.999)
	for _, a := range base.NonClassFloatAttributes(iris) {
		filt.AddAttribute(a)
	}
	filt.Train()
	irisf := base.NewLazilyFilteredInstances(iris, filt)

	fmt.Println(irisf)
}
