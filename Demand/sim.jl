using DataFrames, DataFramesMeta, Distributions, Revise, DebuggingUtilities, CSV, Parameters

string(@__DIR__) in LOAD_PATH || push!(LOAD_PATH, @__DIR__);
using SimModule
const m = SimModule


