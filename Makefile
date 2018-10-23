Slides.pdf: Slides.md
	pandoc -t beamer $< -o $@

images: images/golearn_naive.png \
	images/golearn_linear_models.png \
	images/golearn_trees.png \
	images/golearn_perceptron.png \
	images/jbrukh_bayesian.png \
	images/sajari_regression.png \
	images/campoy_mat.png \
	images/kniren_gota_dataframe.png \
	images/kniren_gota_series.png \
	images/goml_linear.png \
	images/golearn_knn.png

images/goml_linear.png:
	godepgraph github.com/cdipaolo/goml/linear | dot -Tpng -o $@

images/golearn_linear_models.png:
	godepgraph github.com/sjwhitworth/golearn/linear_models | dot -Tpng -o $@

images/golearn_knn.png:
	godepgraph github.com/sjwhitworth/golearn/knn | dot -Tpng -o $@

images/golearn_naive.png:
	godepgraph github.com/sjwhitworth/golearn/naive | dot -Tpng -o $@

images/golearn_trees.png:
	godepgraph github.com/sjwhitworth/golearn/trees | dot -Tpng -o $@

images/golearn_perceptron.png:
	godepgraph github.com/sjwhitworth/golearn/perceptron | dot -Tpng -o $@

images/jbrukh_bayesian.png:
	godepgraph github.com/jbrukh/bayesian | dot -Tpng -o $@

images/sajari_regression.png:
	godepgraph github.com/sajari/regression | dot -Tpng -o $@

images/campoy_mat.png:
	godepgraph github.com/campoy/mat | dot -Tpng -o $@

images/kniren_gota_dataframe.png:
	godepgraph github.com/kniren/gota/dataframe | dot -Tpng -o $@

images/kniren_gota_series.png:
	godepgraph github.com/kniren/gota/series | dot -Tpng -o $@

clean:
	rm -f Slides.pdf

