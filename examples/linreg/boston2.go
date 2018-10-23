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
	regressand := df.Col("medv").Float()

	// Prepare model.
	r := new(regression.Regression)
	r.SetObserved("medv")
	r.SetVar(0, "crim")
	r.SetVar(1, "zn")
	r.SetVar(2, "indus")
	r.SetVar(3, "chas")
	r.SetVar(4, "nox")
	r.SetVar(5, "rm")
	r.SetVar(6, "age")
	r.SetVar(7, "dis")
	r.SetVar(8, "rad")
	r.SetVar(9, "tax")
	r.SetVar(10, "ptratio")
	r.SetVar(11, "b")
	r.SetVar(12, "lstat")

	// Add data points.
	for i, regr := range regressand {
		features := make([]float64, 13)
		for j := 0; j < 13; j++ {
			features[j] = df.Elem(i, j).Float()
		}
		r.Train(regression.DataPoint(regr, features))
	}

	r.Run()

	// Results.
	fmt.Printf("Regression formula: %v\n", r.Formula)

	fmt.Printf("R2: %v\n", r.R2)
	fmt.Printf("Variance observed: %v\n", r.Varianceobserved)
	fmt.Printf("Variance predicted: %v\n", r.VariancePredicted)

}
