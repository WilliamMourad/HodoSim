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

	analysisManager->CreateH1("ScintOpticalPhotonsEnergy", "Scint Optical Photons Energy (eV)", 1000, 2.2, 3.3);
	analysisManager->CreateH1("ScintOpticalPhotonsTime", "Scint Optical Photons Time (ns)", 1000, 0, 30);
	analysisManager->CreateH2("ScintOpticalPhotonsSpread", "Scint Optical Photons Spread; X (mm); Y (mm)", 100, -40, 40, 100, -40, 40);
	analysisManager->CreateH1("OpticalPhotonsReflections0", "Optical Photons Reflections", 1000, 0, 1000);
	analysisManager->CreateH1("OpticalPhotonsReflections1", "Optical Photons Reflections", 1000, 0, 1000);
	analysisManager->CreateH1("OpticalPhotonsReflections2", "Optical Photons Reflections", 1000, 0, 1000);
	analysisManager->CreateH1("OpticalPhotonsReflections3", "Optical Photons Reflections", 1000, 0, 1000);

	analysisManager->CreateNtuple("PerEventCollectedData", "Per-Event Collected Data");
	analysisManager->CreateNtupleDColumn("EventID");
	analysisManager->CreateNtupleDColumn("ScintOPsCollected0");
	analysisManager->CreateNtupleDColumn("ScintOPsCollected1");
	analysisManager->CreateNtupleDColumn("ScintOPsCollected2");
	analysisManager->CreateNtupleDColumn("ScintOPsCollected3");
	analysisManager->CreateNtupleDColumn("CerOPsCollected0");
	analysisManager->CreateNtupleDColumn("CerOPsCollected1");
	analysisManager->CreateNtupleDColumn("CerOPsCollected2");
	analysisManager->CreateNtupleDColumn("CerOPsCollected3");
	analysisManager->CreateNtupleDColumn("ScintTotalEdep");
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