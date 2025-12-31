"use client";

import Papa from "papaparse";
import { ChangeEvent, useMemo, useState } from "react";
import SVM from "ml-svm";

type ParsedRow = Record<string, string>;
type KernelKey = "LINEAR" | "POLY" | "RBF" | "SIGMOID";
type SVMKernel = "linear" | "polynomial" | "rbf" | "sigmoid";

interface TrainedModel {
  label: string;
  svm: InstanceType<typeof SVM>;
}

const SAMPLE_DATA = `sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
5.8,4.0,1.2,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica`;

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 4,
});

const KERNEL_MAP: Record<KernelKey, SVMKernel> = {
  LINEAR: "linear",
  POLY: "polynomial",
  RBF: "rbf",
  SIGMOID: "sigmoid",
};

function parseFeatureValue(value: string) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : NaN;
}

function resolveKernelOptions(
  kernel: KernelKey,
  gammaValue: number | null,
  dimension: number,
) {
  if (kernel === "RBF") {
    const effectiveGamma = gammaValue ?? 1 / Math.max(1, dimension);
    const sigma = Math.sqrt(1 / (2 * effectiveGamma));
    return { sigma };
  }

  if (kernel === "POLY") {
    return { degree: 3, constant: 1, multiplier: 1 };
  }

  if (kernel === "SIGMOID") {
    return { constant: 1, multiplier: 1 };
  }

  return undefined;
}

export default function Home() {
  const [parsedRows, setParsedRows] = useState<ParsedRow[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [labelColumn, setLabelColumn] = useState<string>("");
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("Ready for dataset");
  const [trainingAccuracy, setTrainingAccuracy] = useState<number | null>(null);
  const [models, setModels] = useState<TrainedModel[] | null>(null);
  const [predictionInput, setPredictionInput] = useState<Record<string, string>>(
    {},
  );
  const [prediction, setPrediction] = useState<string>("");
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [gamma, setGamma] = useState<number | null>(null);
  const [cost, setCost] = useState<number>(1);
  const [kernel, setKernel] = useState<KernelKey>("RBF");

  const datasetSummary = useMemo(() => {
    if (!parsedRows.length) return null;
    return {
      rows: parsedRows.length,
      columns: columns.length,
    };
  }, [parsedRows, columns]);

  const handleCSVContent = (csv: string) => {
    const result = Papa.parse<ParsedRow>(csv, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
    });

    if (result.errors.length) {
      setTrainingError(
        `CSV parsing error: ${result.errors[0]?.message ?? "Unknown issue"}`,
      );
      return;
    }

    const rows = result.data.filter(
      (row) => Object.values(row).some((value) => value !== ""),
    );
    const detectedColumns = result.meta.fields ?? [];
    if (!rows.length || !detectedColumns.length) {
      setTrainingError("No rows or columns detected in dataset.");
      return;
    }

    const defaultLabel = detectedColumns[detectedColumns.length - 1];
    const initialFeatures = detectedColumns.filter(
      (column) => column !== defaultLabel,
    );

    setParsedRows(rows);
    setColumns(detectedColumns);
    setLabelColumn(defaultLabel);
    setFeatureColumns(initialFeatures);
    setTrainingError(null);
    setPredictionInput(
      Object.fromEntries(initialFeatures.map((column) => [column, ""])),
    );
    setStatus("Dataset loaded");
    setTrainingAccuracy(null);
    setModels(null);
    setPrediction("");
  };

  const handleFileUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = ({ target }) => {
      const csvContent = target?.result;
      if (typeof csvContent === "string") {
        handleCSVContent(csvContent);
      } else {
        setTrainingError("Unsupported file encoding.");
      }
    };
    reader.readAsText(file);
  };

  const handleFeatureToggle = (column: string) => {
    setFeatureColumns((current) => {
      const isSelected = current.includes(column);
      const updated = isSelected
        ? current.filter((item) => item !== column)
        : [...current, column];

      setPredictionInput((previous) => {
        if (isSelected) {
          const next = { ...previous };
          delete next[column];
          return next;
        }
        return { ...previous, [column]: "" };
      });

      setModels(null);
      setTrainingAccuracy(null);
      setPrediction("");

      return updated;
    });
  };

  const computeFeaturesAndLabels = () => {
    if (!parsedRows.length || !featureColumns.length || !labelColumn) {
      throw new Error("Dataset, features, or label configuration missing.");
    }

    const features: number[][] = [];
    const labels: string[] = [];

    for (const row of parsedRows) {
      const featureVector = featureColumns.map((column) => {
        const value = parseFeatureValue(row[column]);
        if (!Number.isFinite(value)) {
          throw new Error(`Column "${column}" contains non-numeric values.`);
        }
        return value;
      });
      features.push(featureVector);
      labels.push(String(row[labelColumn]));
    }

    return { features, labels };
  };

  const trainModel = async () => {
    try {
      const { features, labels } = computeFeaturesAndLabels();
      if (!features.length) {
        throw new Error("No training data available.");
      }

      const uniqueLabels = Array.from(new Set(labels));
      if (uniqueLabels.length < 2) {
        throw new Error("At least two unique labels are required.");
      }

      const dimension = features[0]?.length ?? 0;
      if (dimension === 0) {
        throw new Error("Select at least one feature column.");
      }

      const kernelType = KERNEL_MAP[kernel];
      const kernelOptions = resolveKernelOptions(kernel, gamma, dimension);

      const trainedModels: TrainedModel[] = [];
      setStatus("Training…");

      await new Promise<void>((resolve) => {
        setTimeout(() => {
          for (const targetLabel of uniqueLabels) {
            const binaryLabels = labels.map((value) =>
              value === targetLabel ? 1 : -1,
            );
            const svm = new SVM({
              C: cost,
              tol: 1e-4,
              maxPasses: 10,
              maxIterations: 10000,
              kernel: kernelType,
              ...(kernelOptions ? { kernelOptions } : {}),
            });
            svm.train(features, binaryLabels);
            trainedModels.push({ label: targetLabel, svm });
          }
          resolve();
        }, 0);
      });

      const predictions = features.map((vector) => {
        let bestLabel = "";
        let bestScore = -Infinity;
        for (const model of trainedModels) {
          const margin = model.svm.marginOne(vector);
          const score = Number.isFinite(margin)
            ? margin
            : model.svm.predictOne(vector);
          if (score > bestScore) {
            bestScore = score;
            bestLabel = model.label;
          }
        }
        return bestLabel;
      });

      const correct = predictions.reduce(
        (count, value, index) => count + (value === labels[index] ? 1 : 0),
        0,
      );

      setModels(trainedModels);
      setTrainingAccuracy(correct / predictions.length);
      setStatus("Model trained");
      setPrediction("");
      setTrainingError(null);
    } catch (error) {
      setStatus("Training failed");
      setTrainingError(
        error instanceof Error ? error.message : "Training failed unexpectedly.",
      );
      setModels(null);
    }
  };

  const handlePredict = () => {
    if (!models || !models.length || !featureColumns.length) {
      setPrediction("Train a model before predicting.");
      return;
    }

    try {
      const vector = featureColumns.map((column) => {
        const rawValue = predictionInput[column];
        const parsed = parseFeatureValue(rawValue);
        if (!Number.isFinite(parsed)) {
          throw new Error(`Prediction input "${column}" must be numeric.`);
        }
        return parsed;
      });

      let bestLabel = "";
      let bestScore = -Infinity;
      for (const model of models) {
        const margin = model.svm.marginOne(vector);
        const score = Number.isFinite(margin)
          ? margin
          : model.svm.predictOne(vector);
        if (score > bestScore) {
          bestScore = score;
          bestLabel = model.label;
        }
      }

      setPrediction(bestLabel);
    } catch (error) {
      setPrediction(
        error instanceof Error ? error.message : "Prediction failed.",
      );
    }
  };

  return (
    <main className="min-h-screen bg-slate-950">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-10 px-6 py-14">
        <section className="rounded-3xl border border-slate-800 bg-slate-900/60 p-8 shadow-xl shadow-slate-950/60">
          <h1 className="text-3xl font-semibold tracking-tight text-white">
            Support Vector Machine Playground
          </h1>
          <p className="mt-3 max-w-2xl text-sm text-slate-300">
            Upload a CSV dataset, choose the target column, tweak hyper-parameters,
            train an SVM model, and run future predictions directly in the browser.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-4 text-xs text-slate-400">
            <span className="rounded-full border border-slate-700 px-3 py-1 uppercase tracking-[0.12em]">
              Status: {status}
            </span>
            {datasetSummary && (
              <span className="rounded-full border border-slate-700 px-3 py-1 uppercase tracking-[0.12em]">
                Rows: {datasetSummary.rows} — Columns: {datasetSummary.columns}
              </span>
            )}
            {trainingAccuracy !== null && (
              <span className="rounded-full border border-emerald-700/80 bg-emerald-500/10 px-3 py-1 uppercase tracking-[0.12em] text-emerald-300">
                Training Accuracy: {numberFormatter.format(trainingAccuracy * 100)}%
              </span>
            )}
          </div>
        </section>

        <section className="grid gap-6 md:grid-cols-[1fr_minmax(0,380px)]">
          <article className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">
            <h2 className="text-lg font-semibold text-white">1. Load Dataset</h2>
            <p className="mt-2 text-sm text-slate-300">
              Provide a CSV file with column headers. The last column becomes the label
              by default, and all preceding columns are treated as features.
            </p>
            <div className="mt-5 flex flex-col gap-3">
              <label className="flex flex-col gap-2 text-sm text-slate-200">
                <span className="font-medium">Upload CSV</span>
                <input
                  type="file"
                  accept=".csv,text/csv"
                  onChange={handleFileUpload}
                  className="block w-full rounded-xl border border-slate-700 bg-slate-800/60 px-3 py-2 text-sm text-slate-100 file:mr-4 file:rounded-lg file:border-0 file:bg-indigo-500 file:px-4 file:py-2 file:text-indigo-50 hover:file:bg-indigo-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-500"
                />
              </label>
              <button
                onClick={() => handleCSVContent(SAMPLE_DATA)}
                className="inline-flex w-fit items-center justify-center rounded-xl bg-indigo-500 px-4 py-2 text-sm font-medium text-indigo-50 transition hover:bg-indigo-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-300"
              >
                Load Iris Sample
              </button>
            </div>

            {columns.length > 0 && (
              <div className="mt-8 space-y-6">
                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-[0.24em] text-slate-400">
                    Label Column
                  </h3>
                  <select
                    value={labelColumn}
                    onChange={(event) => {
                      const nextLabel = event.target.value;
                      setLabelColumn(nextLabel);
                      const nextFeatures = columns.filter(
                        (column) => column !== nextLabel,
                      );
                      setFeatureColumns(nextFeatures);
                      setPredictionInput(
                        Object.fromEntries(
                          nextFeatures.map((column) => [column, ""]),
                        ),
                      );
                      setTrainingAccuracy(null);
                      setModels(null);
                      setPrediction("");
                    }}
                    className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-800/60 px-3 py-2 text-sm text-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-500"
                  >
                    {columns.map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-[0.24em] text-slate-400">
                    Feature Columns
                  </h3>
                  <div className="mt-3 grid gap-2 sm:grid-cols-2">
                    {columns
                      .filter((column) => column !== labelColumn)
                      .map((column) => {
                        const selected = featureColumns.includes(column);
                        return (
                          <button
                            key={column}
                            onClick={() => handleFeatureToggle(column)}
                            className={`flex items-center justify-between rounded-xl border px-3 py-2 text-left text-sm transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-500 ${
                              selected
                                ? "border-indigo-400 bg-indigo-500/20 text-indigo-100"
                                : "border-slate-700 bg-slate-800/60 text-slate-200 hover:border-indigo-400/60 hover:text-indigo-100"
                            }`}
                          >
                            <span>{column}</span>
                            <span
                              className={`rounded-full px-2 py-0.5 text-[10px] uppercase tracking-[0.2em] ${
                                selected
                                  ? "bg-indigo-400/30 text-indigo-200"
                                  : "bg-slate-700/60 text-slate-300"
                              }`}
                            >
                              {selected ? "Included" : "Off"}
                            </span>
                          </button>
                        );
                      })}
                  </div>
                </div>
              </div>
            )}
          </article>

          <article className="flex h-full flex-col rounded-3xl border border-slate-800 bg-slate-900/80 p-6">
            <h2 className="text-lg font-semibold text-white">
              2. Configure & Train
            </h2>
            <div className="mt-5 space-y-4">
              <label className="block text-sm text-slate-300">
                <span className="font-medium text-slate-200">Kernel</span>
                <select
                  value={kernel}
                  onChange={(event) => {
                    setKernel(event.target.value as KernelKey);
                    setTrainingAccuracy(null);
                    setModels(null);
                    setPrediction("");
                  }}
                  className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-800/60 px-3 py-2 text-sm text-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-500"
                >
                  <option value="LINEAR">Linear</option>
                  <option value="POLY">Polynomial</option>
                  <option value="RBF">Radial Basis (RBF)</option>
                  <option value="SIGMOID">Sigmoid</option>
                </select>
              </label>

              <label className="block text-sm text-slate-300">
                <span className="font-medium text-slate-200">Cost (C)</span>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  value={cost}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    setCost(Number.isFinite(value) && value > 0 ? value : 1);
                    setTrainingAccuracy(null);
                    setModels(null);
                    setPrediction("");
                  }}
                  className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-800/60 px-3 py-2 text-sm text-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-500"
                />
              </label>

              <label className="block text-sm text-slate-300">
                <span className="font-medium text-slate-200">
                  Gamma (auto if blank)
                </span>
                <input
                  type="number"
                  step="0.01"
                  value={gamma ?? ""}
                  onChange={(event) => {
                    const raw = event.target.value;
                    if (!raw.length) {
                      setGamma(null);
                      setTrainingAccuracy(null);
                      setModels(null);
                      setPrediction("");
                      return;
                    }
                    const value = Number(raw);
                    setGamma(Number.isFinite(value) && value > 0 ? value : null);
                    setTrainingAccuracy(null);
                    setModels(null);
                    setPrediction("");
                  }}
                  placeholder="Auto"
                  className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-800/60 px-3 py-2 text-sm text-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-500"
                />
              </label>

              <button
                onClick={trainModel}
                disabled={!featureColumns.length || !labelColumn}
                className="mt-4 inline-flex w-full items-center justify-center rounded-xl bg-emerald-500 px-4 py-2 text-sm font-semibold text-emerald-50 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-300"
              >
                Train SVM Model
              </button>

              {trainingError && (
                <p className="rounded-xl border border-rose-500/60 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                  {trainingError}
                </p>
              )}
            </div>

            <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/80 p-4 text-sm text-slate-300">
              <h3 className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
                Model Notes
              </h3>
              <ul className="mt-3 space-y-2 text-xs leading-relaxed text-slate-400">
                <li>
                  All SVM training runs locally in your browser using pure JavaScript.
                </li>
                <li>
                  Ensure the selected feature columns contain numeric values only.
                </li>
                <li>
                  Gamma defaults to 1 / number of features when left blank (RBF kernel).
                </li>
              </ul>
            </div>
          </article>
        </section>

        <section className="rounded-3xl border border-slate-800 bg-slate-900/90 p-6">
          <h2 className="text-lg font-semibold text-white">
            3. Forecast Future Outcomes
          </h2>
          <p className="mt-2 text-sm text-slate-300">
            Provide feature values to generate a prediction with the trained model.
          </p>

          <div className="mt-6 grid gap-4 md:grid-cols-2">
            {featureColumns.map((column) => (
              <label
                key={column}
                className="flex flex-col gap-2 text-sm text-slate-200"
              >
                <span className="font-medium">{column}</span>
                <input
                  type="number"
                  value={predictionInput[column] ?? ""}
                  onChange={(event) =>
                    setPredictionInput((previous) => ({
                      ...previous,
                      [column]: event.target.value,
                    }))
                  }
                  className="rounded-xl border border-slate-700 bg-slate-800/60 px-3 py-2 text-sm text-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-500"
                  placeholder="Enter value"
                />
              </label>
            ))}
          </div>

          <div className="mt-6 flex flex-col gap-3 md:flex-row md:items-center">
            <button
              onClick={handlePredict}
              disabled={!models}
              className="inline-flex items-center justify-center rounded-xl bg-indigo-500 px-4 py-2 text-sm font-semibold text-indigo-50 transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-300"
            >
              Predict
            </button>
            {prediction && (
              <span className="rounded-xl border border-slate-800 bg-slate-950/60 px-4 py-2 text-sm text-slate-100">
                Prediction:{" "}
                <span className="font-semibold text-indigo-300">{prediction}</span>
              </span>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
