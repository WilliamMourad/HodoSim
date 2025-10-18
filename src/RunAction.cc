#include "RunAction.hh"

#include "G4EmCalculator.hh"
#include "G4SystemOfUnits.hh"

RunAction::RunAction(RunActionParameters runActionParameters)
{
	_runActionParameters = runActionParameters;

	timer = new G4Timer();

	// Print the energy cuts corresponding to the range cuts.
	// I need these values to evaluate the restricted Landau/Vavilov edep distribution.
	// I should put this in a custom Geant4 command that can be called on demand
	// but to keep things simple, for now I will just leave it here (expect some repeated output in MT mode).
	if (_runActionParameters.enableCuts)
	{
		G4EmCalculator calc;
		auto Tcut_scint = calc.ComputeEnergyCutFromRangeCut(30 * um, "e-", "G4_PLASTIC_SC_VINYLTOLUENE");
		auto Tcut_si = calc.ComputeEnergyCutFromRangeCut(10 * um, "e-", "G4_Si");
		auto Tcut_al = calc.ComputeEnergyCutFromRangeCut(2 * um, "e-", "G4_Al");

		G4cout << "Tcut (MeV): scint=" << Tcut_scint / MeV
			<< " si=" << Tcut_si / MeV
			<< " al=" << Tcut_al / MeV << G4endl;
	}

	analysisManager = G4AnalysisManager::Instance();
	analysisManager->Reset();
	analysisManager->SetVerboseLevel(1);
	analysisManager->SetNtupleMerging(true);

	// Create Ntuples and histograms here using analysisManager
	// I need to figure out how to parametrize this better, probably passing a struct to the RunAction constructor
	analysisManager->CreateH1("ScintOpticalPhotonsEnergy", "Scint Optical Photons Energy (eV)", 1000, 2.2, 3.3);
	analysisManager->CreateH1("CerOpticalPhotonsEnergy", "Cer Optical Photons Energy (eV)", 100, 1, 10);			// not fundamental, can be removed to save space
	analysisManager->CreateH1("ScintOpticalPhotonsTime", "Scint Optical Photons Time (ns)", 1000, 0, 30);
	analysisManager->CreateH1("CerOpticalPhotonsTime", "Cer Optical Photons Time (ns)", 10, 0, 1);					// not fundamental, can be removed to save space

	analysisManager->CreateH2("ScintOpticalPhotonsSpread", "Scint Optical Photons Spread; X (mm); Y (mm)", 100, -40, 40, 100, -40, 40);	// not fundamental, can be removed to save space
	analysisManager->CreateH2("CerOpticalPhotonsSpread", "Cer Optical Photons Spread; X (mm); Y (mm)", 1000, -40, 40, 1000, -40, 40);		// not fundamental, can be removed to save space

	analysisManager->CreateH1("OpticalPhotonsReflections", "Optical Photons Reflections", 100, 0, 100);

	analysisManager->CreateNtuple("PerEventCollectedData", "Per-Event Collected Data");
	analysisManager->CreateNtupleDColumn("EventID");
	analysisManager->CreateNtupleDColumn("ScintOPsCollected");
	analysisManager->CreateNtupleDColumn("CerOPsCollected");
	analysisManager->CreateNtupleDColumn("ScintTotalEdep");
	analysisManager->CreateNtupleDColumn("SiPMTotalEdep");
	analysisManager->CreateNtupleDColumn("CoatingTotalEdep");
	analysisManager->CreateNtupleDColumn("MuPathLength");
	analysisManager->CreateNtupleDColumn("MuonHitX");
	analysisManager->CreateNtupleDColumn("MuonHitY");
	analysisManager->FinishNtuple();
}

RunAction::~RunAction()
{
	delete timer;
}

void RunAction::BeginOfRunAction(const G4Run* run)
{
	timer->Start();

	analysisManager->OpenFile("output.root");
	
	// Reset ntuple
	// analysisManager->Reset();
}

void RunAction::EndOfRunAction(const G4Run* run)
{
	analysisManager->Write();
	analysisManager->CloseFile(false);

	timer->Stop();
}