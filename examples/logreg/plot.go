package main

import (
	"log"

	"github.com/cdipaolo/goml/base"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func main() {
	log.Println("loading data")

	xTrain, yTrain, err := base.LoadDataFromCSV("train.csv")
	if err != nil {
		log.Fatal(err)
	}
	log.Println("plotting data to exams.png")
	if err := plotData(xTrain, yTrain); err != nil {
		log.Fatal(err)
	}
}

func plotData(xTest [][]float64, yTest []float64) error {
	p, err := plot.New()
	if err != nil {
		return err
	}
	p.Title.Text = "Exam Results"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.X.Max = 120
	p.Y.Max = 120

	positives := make(plotter.XYs, len(yTest))
	negatives := make(plotter.XYs, len(yTest))
	for i := range xTest {
		if yTest[i] == 1.0 {
			positives[i].X = xTest[i][0]
			positives[i].Y = xTest[i][1]
		}
		if yTest[i] == 0.0 {
			negatives[i].X = xTest[i][0]
			negatives[i].Y = xTest[i][1]
		}
	}

	if err := plotutil.AddScatters(p, "Negatives", negatives, "Positives", positives); err != nil {
		return err
	}
	if err := p.Save(10*vg.Inch, 10*vg.Inch, "exams.png"); err != nil {
		return err
	}
	return nil
}
