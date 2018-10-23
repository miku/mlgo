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

	"github.com/sajari/regression"
)

func main() {
	r := new(regression.Regression)
	r.SetObserved("Murders per annum per 1,000,000 inhabitants")
	r.SetVar(0, "Inhabitants")
	r.SetVar(1, "Percent with incomes below $5000")
	r.SetVar(2, "Percent unemployed")
	r.Train(
		regression.DataPoint(11.2, []float64{587000, 16.5, 6.2}),
		regression.DataPoint(13.4, []float64{643000, 20.5, 6.4}),
		regression.DataPoint(40.7, []float64{635000, 26.3, 9.3}),
		regression.DataPoint(5.3, []float64{692000, 16.5, 5.3}),
		regression.DataPoint(24.8, []float64{1248000, 19.2, 7.3}),
		regression.DataPoint(12.7, []float64{643000, 16.5, 5.9}),
		regression.DataPoint(20.9, []float64{1964000, 20.2, 6.4}),
		regression.DataPoint(35.7, []float64{1531000, 21.3, 7.6}),
		regression.DataPoint(8.7, []float64{713000, 17.2, 4.9}),
		regression.DataPoint(9.6, []float64{749000, 14.3, 6.4}),
		regression.DataPoint(14.5, []float64{7895000, 18.1, 6}),
		regression.DataPoint(26.9, []float64{762000, 23.1, 7.4}),
		regression.DataPoint(15.7, []float64{2793000, 19.1, 5.8}),
		regression.DataPoint(36.2, []float64{741000, 24.7, 8.6}),
		regression.DataPoint(18.1, []float64{625000, 18.6, 6.5}),
		regression.DataPoint(28.9, []float64{854000, 24.9, 8.3}),
		regression.DataPoint(14.9, []float64{716000, 17.9, 6.7}),
		regression.DataPoint(25.8, []float64{921000, 22.4, 8.6}),
		regression.DataPoint(21.7, []float64{595000, 20.2, 8.4}),
		regression.DataPoint(25.7, []float64{3353000, 16.9, 6.7}),
	)
	r.Run()

	fmt.Printf("Regression formula:\n%v\n", r.Formula)
	fmt.Printf("Regression:\n%s\n", r)
}
