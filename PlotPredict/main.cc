#define _USE_MATH_DEFINES

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLegendEntry.h>
#include <TPaveText.h>
#include <TMarker.h>
#include <TBox.h>

#include <cmath>
#include <string>
#include <filesystem>
#include <iostream>
#include <onnxruntime_cxx_api.h>


#pragma region Utils

void logMessage(const std::string& msg, bool skip=false) {

	auto prefix = skip ? "" : "[PlotPredict] ";
    std::cout << prefix << msg << std::endl;
}

void SaveBoth(TCanvas* c, const std::string& outdir, const std::string& name, const bool save_pdf = false) {
    c->SaveAs((outdir + "/" + name + ".png").c_str());
    if (save_pdf) c->SaveAs((outdir + "/" + name + ".pdf").c_str());
}

void PlotGraph(TCanvas* c1, TGraph* g1, TGraph* g2, std::string title, std::string outdir, std::string filename)
{
	// Draw the graphs
    g1->SetTitle("Single Hit Per Event - Beam Reconstruction");
    auto ax = g1->GetXaxis();
    auto ay = g1->GetYaxis();

    ax->SetLimits(-40, 40);
    g1->SetMaximum(40);
    g1->SetMinimum(-40);

    ax->SetTitle("X [mm]");
    ay->SetTitle("Y [mm]");

	ay->SetTitleOffset(1.2);

    g1->SetMarkerStyle(21);
    g1->SetMarkerColorAlpha(kBlue, 0.7);
    g1->SetMarkerSize(0.4);
    g1->Draw("ap");

    g2->SetLineWidth(0);
    g2->SetMarkerStyle(21);
    g2->SetMarkerColorAlpha(kRed, 0.7);
    g2->SetMarkerSize(0.4);
    g2->Draw("p same");

    // Draw the plate
    auto box = new TBox(-25, -25, 25, 25);
    box->SetLineColor(kBlue);
    box->SetLineWidth(3);
    box->SetFillColorAlpha(kBlue, 0.1);
    box->Draw();
    
	// Draw legend
    auto leg = new TLegend(0.10, 0.85, 0.47, 0.90);

	leg->SetNColumns(2);
    leg->SetBorderSize(1);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->SetTextFont(42);
    leg->SetTextSize(0.025);

    auto m1 = new TMarker(0, 0, g1->GetMarkerStyle());
	auto m2 = new TMarker(0, 0, g2->GetMarkerStyle());
    m1->SetMarkerColor(g1->GetMarkerColor());
    m1->SetMarkerSize(1.6);
    m2->SetMarkerColor(kRed - 2);
    m2->SetMarkerSize(1.6);
    leg->AddEntry(m1, "True", "p");
    leg->AddEntry(m2, "Predicted", "p");
	
    leg->Draw();

    // Draw stats box
	// for now very basic, just showing means
    auto legStats = new TLegend(0.47, 0.76, 0.90, 0.90);

    legStats->SetNColumns(2);
    legStats->SetBorderSize(1);
    legStats->SetFillStyle(1001);
    legStats->SetFillColor(kWhite);
    legStats->SetTextFont(42);
    legStats->SetTextSize(0.025);

    legStats->SetMargin(0.02);          
    legStats->SetColumnSeparation(0.06); 
    legStats->SetTextAlign(12);
    legStats->AddEntry((TObject*)0, Form("Counts: %d", g1->GetN()), "");
	legStats->AddEntry((TObject*)0, "", "");
    legStats->AddEntry((TObject*)0, Form("<X>_{T}: %.2f mm", g1->GetMean(1)), "");
    legStats->AddEntry((TObject*)0, Form("<X>_{P}: %.2f mm", g2->GetMean(1)), "");
    legStats->AddEntry((TObject*)0, Form("<Y>_{T}: %.2f mm", g1->GetMean(2)), "");
    legStats->AddEntry((TObject*)0, Form("<Y>_{P}: %.2f mm", g2->GetMean(2)), "");

    legStats->Draw();

    SaveBoth(c1, outdir, filename);
    c1->Clear();

}

#pragma endregion Utils


// This function performs predictions using ONNX Runtime to load a small NN that i created and trained outside of this project.
// # On the model
// The NN model is trained to work with 64 SiPMs and is very lightweight (approx 30k parameters).
// It can predict x,y positions from the SiPM light collection features and it was trained assuming a single particle hit per event.
// The model is stored in the file "model.onnx" inside the working directory (this app wont run without it, you have to manually put it there).

void Predict(const char* output_path, std::vector<std::vector<float>> X, const size_t N, const size_t F, const size_t B)
{
	// This function performs predictions using ONNX Runtime,
	// given the input features X (size F x N), where F is the number of features (64 SiPMs) and N is the number of samples (events).
	// The predictions are done in batches of size B to optimize memory usage and performance.
	// The predicted x and y positions are saved in a ROOT file specified by output_path (so that they can be used for analysis).

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx");
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(4);
    Ort::Session sess(env, ORT_TSTR("model.onnx"), opt);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name = sess.GetInputNameAllocated(0, allocator);
    auto out_name = sess.GetOutputNameAllocated(0, allocator);
    const char* IN = in_name.get();
    const char* OUT = out_name.get();

    TFile fout(output_path, "RECREATE");
    float x_pred, y_pred;
    TTree tout("Prediction", "Prediction");
    tout.Branch("x_pred", &x_pred);
    tout.Branch("y_pred", &y_pred);

    std::vector<float> batch(B * F);
    std::vector<float> outbuf(B * 2);

    for (size_t i = 0; i < N; i += B)
    {
        const size_t bsize = std::min(B, N - i);

        for (size_t j = 0; j < bsize; j++)
        {
            for (size_t f = 0; f < F; f++)
            {
                // Remember that i took np.log1p(X) in python!!!
                batch[j * F + f] = log(1 + X[f][i + j]);
            }
        }
        std::array<int64_t, 2> ishape{ (int64_t)bsize, (int64_t)F };
        Ort::Value in = Ort::Value::CreateTensor<float>(mem, batch.data(), bsize * F, ishape.data(), 2);
        auto out = sess.Run({}, &IN, &in, 1, &OUT, 1);
        float* p = out.front().GetTensorMutableData<float>(); // shape: [n,2]

        for (size_t j = 0; j < bsize; j++)
        {
            x_pred = p[j * 2 + 0];
            y_pred = p[j * 2 + 1];
            tout.Fill();
        }
    }
    tout.Write();
    fout.Close();
}


// This app is an experiment and it is still in a very early stage of development, 
// it lacks a lot of features and optimizations since I made it in a hurry just for quick testing purposes.
// Be careful while using it to avoid unexpected behaviors.
//
// To use it, just compile and run it in a directory where you have:
// - a file named "model.onnx" containing the NN model for predictions
// - a directory named "inputs" containing the input ROOT files with the data to process
// The app will create two directories if they don't exist:
// - "plots": where the output plots will be saved
// - "predictions": where the prediction ROOT files will be saved
int main(int argc, char** argv)
{
	// These will eventually become command line arguments
    const std::string outdir = "plots";
    const std::string indir = "inputs";
	const std::string pred_dir = "predictions";
    const int nsipm = 64;

    #pragma region Files and Directories Checks

    // Check that model file exists
    if (!std::filesystem::exists("model.onnx")) {
		logMessage("Error: model file 'model.onnx' does not exist in working directory.");
		return 1;
    }
	logMessage("Model file found at 'model.onnx'.");

    // Check that input directory exists
    if (!std::filesystem::exists(indir)) {
		logMessage("Error: input dir '" + indir + "' does not exist.");
        return 1;
	}

    // Prepare output directory
    try {
        std::filesystem::create_directories(outdir);
    }
    catch (...) {
		logMessage("Warning: could not create output dir '" + outdir + "'.");
        return 1;
    }

    // Prepare predict directory
    try {
        std::filesystem::create_directories(pred_dir);
    }
    catch (...) {
		logMessage("Warning: could not create predictions dir '" + pred_dir + "'.");
        return 1;
    }

    #pragma endregion Files and Directories Checks


	std::vector<std::string> filenames;
    for (const auto& entry : std::filesystem::directory_iterator(indir)) {
        if (entry.path().extension() == ".root") {
			filenames.push_back(entry.path().string());
			logMessage("Input file found at " + entry.path().string());
        }
    }


    TCanvas* c1 = new TCanvas("c1", "c1", 1000, 1000);
    c1->SetLeftMargin(0.1);
    c1->SetRightMargin(0.1);
    c1->SetTopMargin(0.1);
    c1->SetBottomMargin(0.1);
    c1->SetGridy(1);
    c1->SetGridx(1);
    c1->SetTickx(1);
    c1->SetTicky(1);




	logMessage("\n=================================================", true);
    for (const auto& filename : filenames)
    {

		logMessage("Processing file: " + filename);

		std::string prediction_filename = "/pred_" + std::filesystem::path(filename).filename().string();

        ROOT::RDataFrame df("PerEventCollectedData", filename);
    
        #pragma region Predictions Using ONNX Runtime & NN Model
    
        std::vector<std::string> cols(nsipm);
        for (int i = 0; i < nsipm; i++)
        {
            cols[i] = "ScintOPsCollected" + std::to_string(i);
        }

        // Materialize features as float
        std::vector<std::vector<float>> X(64);
        for (int i = 0; i < nsipm; i++)
        {
            auto v = df.Take<double>(cols[i]);
            X[i].assign(v->begin(), v->end());
        }
    
        const size_t N = X[0].size();
        const size_t F = nsipm;
        const size_t B = N;
    
	    Predict((pred_dir + prediction_filename).c_str(), X, N, F, B);

        #pragma endregion Predictions Using ONNX Runtime & NN Model
	
        ROOT::RDataFrame pdf("Prediction", (pred_dir + prediction_filename).c_str());


	    auto muPosX = df.Take<double>("MuonHitX");
	    auto muPosY = df.Take<double>("MuonHitY");

	    auto predX = pdf.Take<float>("x_pred");
	    auto predY = pdf.Take<float>("y_pred");
        
		logMessage("Predicted " + std::to_string(predX->size()) + " points");

        TGraph* g1 = new TGraph(
            muPosX->size(),
            muPosX->data(),
            muPosY->data()
        );

        TGraph* g2 = new TGraph(
            predX->size(),
            predX->data(),
            predY->data()
        );

        PlotGraph(
            c1,
            g1, g2,
            "Beam Reconstruction",
            outdir,
			std::filesystem::path(filename).stem().string()
        );

        logMessage("-------------------------------------------------", true);
    }

    return 0;
}
