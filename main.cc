#include "G4RunManagerFactory.hh"
#include "G4RunManager.hh"
#include "G4MTRunManager.hh"

// My classes
#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "TrackingAction.hh"
#include "SteppingAction.hh"
#include "ActionInitialization.hh"

// Physics 
#include "G4PhysListFactory.hh"
#include "FTFP_BERT.hh"
#include "G4OpticalPhysics.hh"
#include "G4OpticalParameters.hh"
#include "G4ParallelWorldPhysics.hh"

// Visualization and UI
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "G4UImanager.hh"

#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include <ctime>


int main(int argc, char** argv) {
	
	// Set a different random seed based on current time / user preferences
	CLHEP::HepRandom::setTheSeed(time(NULL));


	// A better implementation could be achieved using a json file to set these parameters (i may consider it for future projects)
	#pragma region Simulation Parameters

	G4bool enableTrackingVerbose = false;
	G4bool enableVis = true;	// enable visualization, set to false when running heavy batch jobs
	G4int threads = 20;			// number of threads to be used in MT mode, ignore in ST mode

	G4bool enableCuts = true; // enable production cuts, to test different responses

	G4String siliconPMSDName = "/SiliconPM";
	G4String scintLVName = "ScintLogic";

	G4String opCName = "OpticalPhotonHitsCollection";

	G4double worldSizeXYZ = 1. * m;
	
	// This is more of an explode factor between the solid components
	// I use it mostly for visualization/debug purposes
	G4double gap = 0 * cm; // must be set to 0 in the final version 

	// The thickness of the reflective coating has yet to be formally established
	// but for the purposes of this project any reasonable value will do
	G4double coatingThickness = 50 * um;
	G4double siPMThickness = 3 * mm;

	// The size of the scintillator has yet to be formally established
	// but for the purposes of this project any reasonable value will do
	BoxGeometry scintGeometry = BoxGeometry{
		50. * mm, 
		50. * mm, 
		3 * mm 
	};

	// I set these values to match those of a BC-412 scintillator 
	// (the full datasheet can be found at https://www.robot-domestici.it/joomla/component/virtuemart/scintillatore-plastico-50--50--5-mm/getfile?file_id=5392)
	ScintillatorProperties scintData = ScintillatorProperties{
		scintGeometry,
		1,			// scalingFactor: lower values reduce the scintillation yield
		8000.,			// scintYield (photon/MeV)
		2.85 * eV,		// waveLengthPeak	(434 nm)
		3.09 * eV,		// waveLengthLeft	(402 nm)
		2.41 * eV,		// waveLengthRight	(514 nm)
		3.3 * ns,		// decayTime
		1.58,			// refractiveIndex
		210. * cm,		// absorptionLength
	};

	// These values are used in the primary generator action
	G4String particleName = "mu-";

	ParticleGunSettings gunSettings = ParticleGunSettings{
		false,							// isActive
		1,								// particleN
		55 * MeV,						// particleEnergy
		{0, 0, -2.5 * cm},				// gunPosition
		{0, 0, 1}						// gunDirection
	};

	GPSSettings gpsSettings = GPSSettings{
		true,							// isActive
		1,								// particleN
		55 * MeV,						// particleMeanEnergy
		0 * MeV,						// particleEnergySigma
		{1. * cm, 1. * cm, -2.5 * cm},	// gpsPosition
		0.6 * cm,						// gpsRadius
		{1, 0, 0},						// gpsRot1
		{0, -1, 0},						// gpsRot2
		0.01,							// beamApertureX
		0.01							// beamApertureY
	};

	#pragma endregion Simulation Parameters


	#pragma region RunManager Definition

	// MT Mode
	G4MTRunManager* runManager = new G4MTRunManager;
	runManager->SetNumberOfThreads(threads);

	// ST Mode
	// G4RunManager* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);

	#pragma endregion RunManager Definition


	#pragma region PhysicsList Definition & Initialization


	G4PhysListFactory factory;
	auto physicsList = factory.GetReferencePhysList("FTFP_BERT_EMZ");

	auto optPhysics = new G4OpticalPhysics();
	auto optParams = G4OpticalParameters::Instance();
	optParams->SetScintTrackSecondariesFirst(true);

	physicsList->RegisterPhysics(optPhysics);
	physicsList->SetVerboseLevel(0);
	
	# pragma endregion PhysicsList Definition & Initialization


	#pragma region DetectorConstruction Definition & Initialization

	DetectorConstruction* detectorConstruction = new DetectorConstruction(
		worldSizeXYZ, 
		scintData, 
		coatingThickness,
		siPMThickness,
		gap,
		siliconPMSDName,
		scintLVName,
		opCName,
		enableCuts
	);

	#pragma endregion DetectorConstruction Definition & Initialization


	#pragma region User Actions Definition

	PrimaryGeneratorActionParameters primaryGeneratorActionParameters = PrimaryGeneratorActionParameters{
		particleName,
		gunSettings,
		gpsSettings
	};
	
	RunActionParameters runActionParameters = RunActionParameters{
		enableCuts
	};
	
	EventActionParameters eventActionParameters = EventActionParameters{ 
		scintLVName,
		siliconPMSDName, 
		opCName,
		G4ThreeVector(
			scintGeometry.sizeX,
			scintGeometry.sizeY,
			scintGeometry.sizeZ
		)
	};

	TrackingActionParameters trackingActionParameters = TrackingActionParameters{};

	SteppingActionParameters steppingActionParameters = SteppingActionParameters{
		scintLVName
	};

	#pragma endregion User Actions Definition
	

	runManager->SetUserInitialization(detectorConstruction);
	runManager->SetUserInitialization(physicsList);
	runManager->SetUserInitialization(new ActionInitialization(
		primaryGeneratorActionParameters,
		runActionParameters,
		eventActionParameters,
		trackingActionParameters,
		steppingActionParameters
	));

	runManager->Initialize();
	
	auto ui = new G4UIExecutive(argc, argv);
	auto UImanager = G4UImanager::GetUIpointer();
	auto visManager = new G4VisExecutive(argc, argv);
	

	UImanager->ApplyCommand("/control/execute macros\\init.mac");

	UImanager->ApplyCommand("/control/execute macros\\build_custom_gui.mac");
	if (enableVis)
	{
		visManager->Initialize();
		UImanager->ApplyCommand("/control/execute macros\\vis.mac");
		UImanager->ApplyCommand("/control/execute macros\\paint_geometry.mac");
	}

	if (enableTrackingVerbose)
	{
		UImanager->ApplyCommand("/tracking/verbose 1");
	}
	else {
		UImanager->ApplyCommand("/tracking/verbose 0");
	}

	ui->SessionStart();
	
	delete ui;
	if (enableVis){ delete visManager; }
	delete runManager;

	return 0;
}
