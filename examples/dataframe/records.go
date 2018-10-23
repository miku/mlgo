package main

import (
	"fmt"

	"github.com/kniren/gota/dataframe"
	"github.com/kniren/gota/series"
)

func main() {

	df := dataframe.LoadRecords(
		[][]string{
			[]string{"A", "B", "C", "D"},
			[]string{"a", "4", "5.1", "true"},
			[]string{"k", "5", "7.0", "true"},
			[]string{"k", "4", "6.0", "true"},
			[]string{"a", "2", "7.1", "false"},
		},
		dataframe.DetectTypes(false),
		dataframe.DefaultType(series.Float),
		dataframe.WithTypes(map[string]series.Type{
			"A": series.String,
			"D": series.Bool,
		}),
	)
	fmt.Println(df)
}
