# Repository

Companion code for a Medium article.

## Business context

Model residuals tell the story of what a forecasting model missed. A solid residual analysis helps validate the model and reveals areas for improvement. This chapter breaks down how to analyze forecast residuals effectively.

Residuals are the gaps between what the model predicts and what actually happens. In a well-functioning time series model, these should behave like white noise --- random, uncorrelated, and normally distributed. If they don't, there's room to refine the model.

A histogram and density plot show how residuals are distributed. A symmetric, bell-shaped curve suggests normality. Deviations hint at skewness or heavy tails. ![image](img/qq_plot.png) A Q-Q plot compares the residual distribution to a normal distribution. If residuals fall along the diagonal line, they follow a normal distribution. Deviations suggest non-normality, which can affect statistical assumptions.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).