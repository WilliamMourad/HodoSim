#include "ParallelWorld.hh"

#include "G4NistManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4VPhysicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Box.hh"

#include "G4SDManager.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4PSEnergyDeposit.hh"
#include "PrimaryMuonFilter.hh"


ParallelWorld::ParallelWorld(const G4String& name, G4double plateThickness, G4double siPMThickness, G4double gap) : G4VUserParallelWorld(name)
{
	_plateThickness = plateThickness;
	_siPMThickenss = siPMThickness;
	_gap = gap;
};

ParallelWorld::~ParallelWorld() {};

void ParallelWorld::Construct()
{
	// I'll keep it very minimal and i'll put all the code in this single method for now.
	// I won't add many comments, since it is pretty straightforward,
	// the idea is to create a ghost copy (vacuum) of the SiPM logical volume
	// to which the MFD will be attached for the edep collection.

	auto* pwWorldPV = GetWorld();
	auto* pwWorldLV = pwWorldPV->GetLogicalVolume();

	auto* vacuum = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");

	auto* siPMLV = G4LogicalVolumeStore::GetInstance()->GetVolume("SiPMLogic");
	auto* siPMBox = dynamic_cast<G4Box*>(siPMLV->GetSolid());

	fGhostSiPMLV = new G4LogicalVolume(
		new G4Box(
			"GhostSiPMSolid",
			siPMBox->GetXHalfLength(),
			siPMBox->GetYHalfLength(),
			siPMBox->GetZHalfLength()
		),
		vacuum,
		"GhostSiPMLogic"
	);

	// this has to match the real SiPM position
	G4ThreeVector ghostSiPMPosition = G4ThreeVector(0, 0, _plateThickness / 2 + _siPMThickenss / 2 + _gap);


	new G4PVPlacement(
		nullptr,
		ghostSiPMPosition,
		fGhostSiPMLV,
		"GhostSiPMPhysical",
		pwWorldLV,
		false,
		0,
		false
	);
};

void ParallelWorld::ConstructSD()
{
	// Just attach a MFD to the ghost SiPM logical volume and register a primitive scorer for edep

	auto* sdManager = G4SDManager::GetSDMpointer();
	auto* muFilter = new PrimaryMuonFilter("PrimaryMuFilter");

	auto* ghostSiPMMFD = new G4MultiFunctionalDetector("GhostSiPMMFD");
	ghostSiPMMFD->RegisterPrimitive(new G4PSEnergyDeposit("Edep"));
	ghostSiPMMFD->SetFilter(muFilter);

	sdManager->AddNewDetector(ghostSiPMMFD);

	fGhostSiPMLV->SetSensitiveDetector(ghostSiPMMFD);
};