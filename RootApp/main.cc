#define _USE_MATH_DEFINES

#include <windows.h> // comment this out on non-windows systems
#include "TApplication.h"
#include "TCanvas.h"
#include "TRootCanvas.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TFile.h"
#include "TROOT.h"
#include "TF1.h"
#include "TFitResult.h"
#include "TFitResultPtr.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TMath.h"


// Fit formula
Double_t FitFormula(Double_t* x, Double_t* par)
{
    // x[0]
    // par[0] is normalization factor
    // par[1] is offset for background (optional)

    // Formula: y = ax + b
    Double_t formula = par[0] * x[0] + par[1];

    return formula;
}


// This is just a simple app to quickly check/save plots of the data directly from the Geant4 gui.
// it can be compiled as a standalone executable and run in the same directory where the output.root file is located.
// Since i developed it in Windows, i compiled it as a Win applicaiton (so that it could be run safely from the Geant4 app without summoning a new terminal each time).
// To compile it on unix systems:
// - replace WinMain with main and remove the windows.h include,
// - remove the WIN32 parameter (when calling add_executable) from the RootApp/CMakeLists.txt file 
// 
// Right now it just saves the histograms collected in the output.root file as images.
// It can be expanded to do more useful things, but i don't plan on adding them for this specific project.
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
// int main(int argc, char** argv)
{
    // Use **argc and **argv to get the command-line arguments
    int argc = __argc;
    char** argv = __argv;

    bool showInInteractiveMode = false;

    // Create the ROOT application
    TApplication app("RootApp", &argc, argv);
    gROOT->Reset();
    gROOT->SetStyle("Plain");

    // Improve visual style
    gStyle->SetOptStat(1111);     // Show statistics box
    gStyle->SetOptFit(1111);      // Show fit parameters
    gStyle->SetPalette(1);        // Use default color palette

    // Open the output ROOT file
    TFile f("output.root");
    if (!f.IsOpen()) {
        MessageBox(NULL, "Failed to open output.root", "Error", MB_ICONERROR);
        return -1;
    }

    // Create a canvas
    TCanvas* canvas = new TCanvas("canvas",
        "MuScinTest",
        1000, 800);

    // Plot Histogram as beautiful columns 
	TH1D* h1 = (TH1D*)f.Get("ScintOpticalPhotonsEnergy");
    if (!h1) {
        MessageBox(NULL, "Failed to retrieve histogram h1", "Error", MB_ICONERROR);
        return -1;
	}
	h1->SetFillColor(kBlue - 10);
	h1->SetLineColor(kBlue + 2);
	h1->SetLineWidth(2);
	h1->Draw("hist");
	h1->GetXaxis()->SetTitle("Energy (eV)");
	h1->GetYaxis()->SetTitle("Counts / Run");
	h1->SetTitle("Scint Optical Photons Energy");
    canvas->Update();

	canvas->SaveAs("ScintOpticalPhotonsEnergy.png");

    TH1D* h2 = (TH1D*)f.Get("CerOpticalPhotonsEnergy");
    if (!h2) {
        MessageBox(NULL, "Failed to retrieve histogram h2", "Error", MB_ICONERROR);
        return -1;
    }
    h2->SetFillColor(kBlue - 10);
    h2->SetLineColor(kBlue + 2);
    h2->SetLineWidth(2);
    h2->Draw("hist");
    h2->GetXaxis()->SetTitle("Energy (eV)");
    h2->GetYaxis()->SetTitle("Counts / Run");
    h2->SetTitle("Cer Optical Photons Energy");
    canvas->Update();

    canvas->SaveAs("CerOpticalPhotonsEnergy.png");

    TH1D* h3 = (TH1D*)f.Get("ScintOpticalPhotonsTime");
    if (!h3) {
        MessageBox(NULL, "Failed to retrieve histogram h3", "Error", MB_ICONERROR);
        return -1;
    }
    h3->SetFillColor(kBlue - 10);
    h3->SetLineColor(kBlue + 2);
    h3->SetLineWidth(2);
    h3->Draw("hist");
    h3->GetXaxis()->SetTitle("Time (ns)");
    h3->GetYaxis()->SetTitle("Counts / Run");
    h3->SetTitle("Scint Optical Photons Time");
    canvas->Update();

    canvas->SaveAs("ScintOpticalPhotonsTime.png");


    TH1D* h4 = (TH1D*)f.Get("CerOpticalPhotonsTime");
    if (!h4) {
        MessageBox(NULL, "Failed to retrieve histogram h4", "Error", MB_ICONERROR);
        return -1;
    }
    h4->SetFillColor(kBlue - 10);
    h4->SetLineColor(kBlue + 2);
    h4->SetLineWidth(2);
    h4->Draw("hist");
    h4->GetXaxis()->SetTitle("Time (ns)");
    h4->GetYaxis()->SetTitle("Counts / Run");
    h4->SetTitle("Cer Optical Photons Time");
    canvas->Update();

    canvas->SaveAs("CerOpticalPhotonsTime.png");

    // plot 2d histogram 
	TH2D* h5 = (TH2D*)f.Get("ScintOpticalPhotonsSpread");
    if (!h5) {
        MessageBox(NULL, "Failed to retrieve histogram h5", "Error", MB_ICONERROR);
        return -1;
	}
	h5->SetTitle("Scint Optical Photons Spread");
	h5->GetXaxis()->SetTitle("X (mm)");
	h5->GetYaxis()->SetTitle("Y (mm)");
	h5->SetMarkerColor(kBlue + 2);
	h5->Draw("COLZ");
	canvas->Update();

	canvas->SaveAs("ScintOpticalPhotonsSpread.png");


	TH2D* h6 = (TH2D*)f.Get("CerOpticalPhotonsSpread");
    if (!h6) {
        MessageBox(NULL, "Failed to retrieve histogram h6", "Error", MB_ICONERROR);
        return -1;
	}
	h6->SetTitle("Cer Optical Photons Spread");
	h6->GetXaxis()->SetTitle("X (mm)");
	h6->GetYaxis()->SetTitle("Y (mm)");
	h6->SetMarkerColor(kBlue + 2);
	h6->Draw("COLZ");
	canvas->Update();

	canvas->SaveAs("CerOpticalPhotonsSpread.png");

    
    TH1D* h7 = (TH1D*)f.Get("OpticalPhotonsReflections");
    if (!h7) {
        MessageBox(NULL, "Failed to retrieve histogram h7", "Error", MB_ICONERROR);
        return -1;
    }
    h7->SetFillColor(kBlue - 10);
    h7->SetLineColor(kBlue + 2);
    h7->SetLineWidth(2);
    h7->Draw("hist");
    h7->GetXaxis()->SetTitle("Reflections");
    h7->GetYaxis()->SetTitle("Frequency");
    h7->SetTitle("Optical Photons Reflections");
    canvas->Update();

    canvas->SaveAs("OpticalPhotonsReflections.png");


    // Connect the canvas's "Closed" signal to terminate the application
    TRootCanvas* rc = (TRootCanvas*)canvas->GetCanvasImp();
    rc->Connect("CloseWindow()", "TApplication", gApplication, "Terminate()");

    if (showInInteractiveMode)
    {
        app.Run();
    }
    return 0;
}
