// Data Set Characteristics:
//
//     :Number of Instances: 506
//
//     :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
//
//     :Attribute Information (in order):
//         - CRIM     per capita crime rate by town
//         - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
//         - INDUS    proportion of non-retail business acres per town
//         - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
//         - NOX      nitric oxides concentration (parts per 10 million)
//         - RM       average number of rooms per dwelling
//         - AGE      proportion of owner-occupied units built prior to 1940
//         - DIS      weighted distances to five Boston employment centres
//         - RAD      index of accessibility to radial highways
//         - TAX      full-value property-tax rate per $10,000
//         - PTRATIO  pupil-teacher ratio by town
//         - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
//         - LSTAT    % lower status of the population
//         - MEDV     Median value of owner-occupied homes in $1000's
//
//     :Missing Attribute Values: None
//
//     :Creator: Harrison, D. and Rubinfeld, D.L.

package main

import (
	"fmt"
	"log"
	"os"

	"github.com/kniren/gota/dataframe"
	"github.com/sajari/regression"
)

func main() {

	f, err := os.Open("BostonHousing.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	df := dataframe.ReadCSV(f)
	if df.Err != nil {
		log.Fatal(df.Err)
	}

	// Get float values per column.
	feature := df.Col("rm").Float()
	regressand := df.Col("medv").Float()

	// Length of data points should match.
	if len(feature) != len(regressand) {
		log.Fatal("unaligned data")
	}

	// Prepare model.
	r := new(regression.Regression)
	r.SetObserved("medv")
	r.SetVar(0, "rm")

	// Add data points.
	for i, f := range feature {
		log.Println(regressand[i], f)
		r.Train(regression.DataPoint(regressand[i], []float64{f}))
	}

	r.Run()

	// Results.
	fmt.Printf("Regression formula: %v\n", r.Formula)

	p, err := r.Predict([]float64{7.0})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(p)
}
