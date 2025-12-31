declare module "ml-svm" {
  export type KernelType = "linear" | "polynomial" | "rbf" | "sigmoid";

  export interface PolynomialKernelOptions {
    degree?: number;
    constant?: number;
    multiplier?: number;
  }

  export interface RbfKernelOptions {
    sigma?: number;
  }

  export interface SigmoidKernelOptions {
    constant?: number;
    multiplier?: number;
  }

  export type KernelOptions =
    | PolynomialKernelOptions
    | RbfKernelOptions
    | SigmoidKernelOptions
    | Record<string, unknown>;

  export interface SVMOptions {
    C?: number;
    tol?: number;
    maxPasses?: number;
    maxIterations?: number;
    kernel?: KernelType;
    kernelOptions?: KernelOptions;
  }

  export default class SVM {
    constructor(options?: SVMOptions);
    train(features: number[][], labels: number[]): void;
    predict(features: number[][]): number[];
    predictOne(feature: number[]): number;
    margin(features: number[][]): number[];
    marginOne(feature: number[]): number;
    supportVectors(): { X: number[][]; y: number[] };
    toJSON(): Record<string, unknown>;
  }
}
