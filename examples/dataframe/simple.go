package main

import (
	"fmt"

	"github.com/kniren/gota/dataframe"
	"github.com/kniren/gota/series"
)

func main() {
	df := dataframe.New(
		series.New([]string{"b", "a"}, series.String, "COL.1"),
		series.New([]int{1, 2}, series.Int, "COL.2"),
		series.New([]float64{3.0, 4.0}, series.Float, "COL.3"),
	)

	fmt.Println(df)
}
