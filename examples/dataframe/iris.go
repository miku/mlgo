package main

import (
	"fmt"
	"log"
	"os"

	"github.com/kniren/gota/dataframe"
	"github.com/kniren/gota/series"
)

func main() {
	df := dataframe.ReadCSV(os.Stdin,
		dataframe.DefaultType(series.String),
		dataframe.DetectTypes(true),
		dataframe.HasHeader(false),
		dataframe.Names("sl", "sw", "pl", "pw", "species"),
		dataframe.NaNValues([]string{"NA", "NaN", "<nil>"}))

	fmt.Println(df)

	// Get a column.
	sl := df.Col("sl")

	// Basic stats.
	fmt.Println(sl)

	// Basic stats.
	mean := func(s series.Series) series.Series {
		floats := s.Float()
		sum := 0.0
		for _, f := range floats {
			sum += f
		}
		return series.Floats(sum / float64(len(floats)))
	}

	// Apply, will not err on non-numeric columns.
	cmean := df.Capply(mean)
	rmean := df.Select([]string{"sl", "sw", "pl", "pw"}).Rapply(mean)

	fmt.Println(cmean)
	fmt.Println(rmean)

	fi := df.Filter(dataframe.F{"sw", ">", 2.0}).
		Filter(dataframe.F{"species", "==", "Iris-setosa"}).
		Select([]string{"pw", "pl"}).
		Subset([]int{0, 2})

	if fi.Err != nil {
		log.Fatal(fi.Err)
	}

	fmt.Println(fi)

	fmt.Println(df.Describe())
}
