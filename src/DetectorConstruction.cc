#include "DetectorConstruction.hh"
#include "SiliconPMSD.hh"
#include "PrimaryMuonFilter.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

#include "G4SDManager.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4PSEnergyDeposit.hh"
#include "G4PSPassageTrackLength.hh"
#include "G4ProductionCuts.hh"

#include "G4SystemOfUnits.hh"


DetectorConstruction::DetectorConstruction(
	G4double worldSizeXYZ, 
	ScintillatorProperties scintData, 
	G4double coatingThickness, 
	G4double siPMThickness, 
	G4double gap,
	G4String siliconPMSDName,
	G4String scintLVName,
	G4String opCName,
	G4bool enableCuts
) : G4VUserDetectorConstruction()
{
	_worldSizeXYZ = worldSizeXYZ;
	_scintData = scintData;
	_coatingThickness = coatingThickness;
	_siPMThickness = siPMThickness; 
	_gap = gap;

	_siliconPMSDName = siliconPMSDName;
	_scintLVName = scintLVName;
	_opCName = opCName;

	_enableCuts = enableCuts;

	nist = G4NistManager::Instance();
}

DetectorConstruction::~DetectorConstruction() {}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
	DefineMaterials();
	return BuildGeometry();
}


// Geometry Definition & Builder
G4VPhysicalVolume* DetectorConstruction::BuildGeometry()
{
	G4double worldSizeXYZ = _worldSizeXYZ;

	G4double plateSizeX = _scintData.geometry.sizeX;
	G4double plateSizeY = _scintData.geometry.sizeY;
	G4double plateThickness = _scintData.geometry.sizeZ;

	G4double coatingThickness = _coatingThickness;

	G4double siPMThickenss = _siPMThickness;

	G4double gap = _gap;

	// Geometry Definition
	#pragma region Geometry Definitions & Placements

		#pragma region World Geometry
	// World 
	G4Box* worldSolid = new G4Box("WorldSolid", worldSizeXYZ / 2, worldSizeXYZ / 2, worldSizeXYZ / 2);
	G4LogicalVolume* worldLogic = new G4LogicalVolume(worldSolid, vacuum, "WorldLogic");

	G4VPhysicalVolume* worldPhysical = new G4PVPlacement(
		nullptr,
		G4ThreeVector(),
		worldLogic,
		"WorldPhysical",
		nullptr,
		false,
		0,
		true
	);
		#pragma endregion World Geometry
	
		#pragma region Scintillator Geometry
	// Scintillator Plate
	G4Box* scintSolid = new G4Box(
		"ScintSolid",
		plateSizeX / 2,
		plateSizeY / 2,
		plateThickness / 2
	);
	G4LogicalVolume* scintLogic = new G4LogicalVolume(scintSolid, scint_material, _scintLVName);
	G4ThreeVector scintPosition = G4ThreeVector(0, 0, 0);

	G4VPhysicalVolume* scintPhysical = new G4PVPlacement(
		nullptr,
		scintPosition,
		scintLogic,
		"ScintPhysical",
		worldLogic,
		false,
		0,
		true
	);

		#pragma endregion Scintillator Geometry

		#pragma region Coating Geometry
	// X-Sides Coating
	G4Box* xCoatings = new G4Box(
		"XCoatSolid",
		plateSizeX / 2,
		coatingThickness / 2,
		plateThickness / 2
	);
	G4LogicalVolume* xCoatingLogic = new G4LogicalVolume(xCoatings, coating_material, "XCoatLogic");
	G4ThreeVector xCoatingTopPosition = G4ThreeVector(0, plateSizeY / 2 + coatingThickness / 2 + gap, 0);
	G4ThreeVector xCoatingBottomPosition = G4ThreeVector(0, -plateSizeY / 2 - coatingThickness / 2 - gap, 0);

	G4VPhysicalVolume* xCoatingTopPhysical = new G4PVPlacement(
		nullptr,
		xCoatingTopPosition,
		xCoatingLogic,
		"XCoatTopPhysical",
		worldLogic,
		false,
		0,
		true
	);

	G4VPhysicalVolume* xCoatingBottomPhysical = new G4PVPlacement(
		nullptr,
		xCoatingBottomPosition,
		xCoatingLogic,
		"XCoatBottomPhysical",
		worldLogic,
		false,
		0,
		true
	);

	// Y-Sides Coating
	G4Box* yCoatings = new G4Box(
		"YCoatSolid",
		coatingThickness / 2,
		plateSizeY / 2,
		plateThickness / 2
	);
	G4LogicalVolume* yCoatingLogic = new G4LogicalVolume(yCoatings, coating_material, "YCoatLogic");
	G4ThreeVector yCoatingLeftPosition = G4ThreeVector(-plateSizeX / 2 - coatingThickness / 2 - gap, 0, 0);
	G4ThreeVector yCoatingRightPosition = G4ThreeVector(plateSizeX / 2 + coatingThickness / 2 + gap, 0, 0);

	G4VPhysicalVolume* yCoatingLeftPhysical = new G4PVPlacement(
		nullptr,
		yCoatingLeftPosition,
		yCoatingLogic,
		"YCoatLeftPhysical",
		worldLogic,
		false,
		0,
		true
	);

	G4VPhysicalVolume* yCoatingRightPhysical = new G4PVPlacement(
		nullptr,
		yCoatingRightPosition,
		yCoatingLogic,
		"YCoatRightPhysical",
		worldLogic,
		false,
		0,
		true
	);

	// Front Coating
	G4Box* frontCoating = new G4Box(
		"FrontCoatSolid",
		plateSizeX / 2,
		plateSizeY / 2,
		coatingThickness / 2
	);
	G4LogicalVolume* frontCoatingLogic = new G4LogicalVolume(frontCoating, coating_material, "FrontCoatLogic");
	G4ThreeVector frontCoatingPosition = G4ThreeVector(0, 0, -plateThickness / 2 - coatingThickness / 2 - gap);

	G4VPhysicalVolume* frontCoatingPhysical = new G4PVPlacement(
		nullptr,
		frontCoatingPosition,
		frontCoatingLogic,
		"FrontCoatPhysical",
		worldLogic,
		false,
		0,
		true
	);

		#pragma endregion Coating Geometry

		#pragma region SiPM Geometry
	// SiPM 
	G4Box* siPMSolid = new G4Box(
		"SiPMSolid",
		plateSizeX / 2,
		plateSizeY / 2,
		siPMThickenss / 2
	);
	G4LogicalVolume* siPMLogic = new G4LogicalVolume(siPMSolid, sipm_material, "SiPMLogic");
	G4ThreeVector siPMPosition = G4ThreeVector(0, 0, plateThickness / 2 + siPMThickenss / 2 + gap);
	
	G4VPhysicalVolume* siPMPhysical = new G4PVPlacement(
		nullptr,
		siPMPosition,
		siPMLogic,
		"SiPMPhysical",
		worldLogic,
		false,
		0,
		true
	);

		#pragma endregion SiPM Geometry
	

	// This was a test for a split of the SiPM into SiPM bulk and SiPM photodetector,
	// it was meant to sample optical properties and edep separately,
	// but in the end i went for a parallel world since with this method i had to sacrifice cerenkov optics.
	// 
	// G4Box* siPMBulkSolid = new G4Box(
	// 	"SiPMBulkSolid",
	// 	plateSizeX / 2,
	// 	plateSizeY / 2,
	// 	siPMThickenss / 2 * 0.9
	// );
	// G4LogicalVolume* siPMBulkLogic = new G4LogicalVolume(siPMBulkSolid, sipm_material, "SiPMBulkLogic");
	// G4ThreeVector siPMBulkPosition = G4ThreeVector(0, 0, plateThickness / 2 + siPMThickenss / 2 * 0.9 + gap);
	// 
	// G4VPhysicalVolume* siPMBulkPhysical = new G4PVPlacement(
	// 	nullptr,
	// 	siPMBulkPosition,
	// 	siPMBulkLogic,
	// 	"SiPMBulkPhysical",
	// 	worldLogic,
	// 	false,
	// 	0,
	// 	true
	// );


	#pragma endregion Geometry Definitions & Placements

	// Boundary Surfaces Definitions
	#pragma region Optical Surfaces Definitions

	// Scintillator <-> Coating Surfaces
		#pragma region Scintillator-Coating Surfaces
	scint_to_coating_top = new G4LogicalBorderSurface(
		"ScintToCoatingTop",
		scintPhysical,
		xCoatingTopPhysical,
		coating_surface
	);

	scint_to_coating_bottom = new G4LogicalBorderSurface(
		"ScintToCoatingBottom",
		scintPhysical,
		xCoatingBottomPhysical,
		coating_surface
	);

	scint_to_coating_left = new G4LogicalBorderSurface(
		"ScintToCoatingLeft",
		scintPhysical,
		yCoatingLeftPhysical,
		coating_surface
	);

	scint_to_coating_right = new G4LogicalBorderSurface(
		"ScintToCoatingRight",
		scintPhysical,
		yCoatingRightPhysical,
		coating_surface
	);

	scint_to_coating_front = new G4LogicalBorderSurface(
		"ScintToCoatingFront",
		scintPhysical,
		frontCoatingPhysical,
		coating_surface
	);

	coating_top_to_scint = new G4LogicalBorderSurface(
		"CoatingTopToScint",
		xCoatingTopPhysical,
		scintPhysical,
		scint_surface
	);

	coating_bottom_to_scint = new G4LogicalBorderSurface(
		"CoatingBottomToScint",
		xCoatingBottomPhysical,
		scintPhysical,
		scint_surface
	);

	coating_left_to_scint = new G4LogicalBorderSurface(
		"CoatingLeftToScint",
		yCoatingLeftPhysical,
		scintPhysical,
		scint_surface
	);

	coating_right_to_scint = new G4LogicalBorderSurface(
		"CoatingRightToScint",
		yCoatingRightPhysical,
		scintPhysical,
		scint_surface
	);

	coating_front_to_scint = new G4LogicalBorderSurface(
		"CoatingFrontToScint",
		frontCoatingPhysical,
		scintPhysical,
		scint_surface
	);

		#pragma endregion Scintillator-Coating Surfaces
	
	// Scintillator <-> SiPM Surfaces
		#pragma region Scintillator-SiPM Surfaces
	
	scint_to_sipm = new G4LogicalBorderSurface(
		"ScintToSiPM",
		scintPhysical,
		siPMPhysical,
		sipm_surface
	);

	sipm_to_scint = new G4LogicalBorderSurface(
		"SiPMToScint",
		siPMPhysical,
		scintPhysical,
		scint_surface
	);
	
		#pragma endregion Scintillator-SiPM Surfaces

	#pragma endregion Optical Surfaces Definitions

	// Production Cuts Definitions
	#pragma region Cuts

	if (_enableCuts)
	{
		auto makeRegion = [](const G4String name, G4LogicalVolume* lv, G4double cutGam, G4double cutEm, G4double cutEp){
			auto* reg = new G4Region(name);
			reg->AddRootLogicalVolume(lv);
			auto* cuts = new G4ProductionCuts();
			cuts->SetProductionCut(cutGam, "gamma");
			cuts->SetProductionCut(cutEm, "e-");
			cuts->SetProductionCut(cutEp, "e+");
			reg->SetProductionCuts(cuts);
		};

		// I followed a 1/5 or 1/10 rule of thumb for the cuts
		makeRegion("ScintRegion", scintLogic, 50 * um, 30 * um, 30 * um);
		makeRegion("SiPMRegion", siPMLogic, 20 * um, 10 * um, 10 * um);
		makeRegion("XCoatingRegion", xCoatingLogic, 5 * um, 2 * um, 2 * um);
		makeRegion("YCoatingRegion", yCoatingLogic, 5 * um, 2 * um, 2 * um);
		makeRegion("FrontCoatingRegion", frontCoatingLogic, 5 * um, 2 * um, 2 * um);
	}

	#pragma endregion Cuts
	
	return worldPhysical;
}


// Materials Definitions
void DetectorConstruction::DefineMaterials()
{
	vacuum = nist->FindOrBuildMaterial("G4_Galactic");
	air = nist->FindOrBuildMaterial("G4_AIR");

	// Group material definitions for clarity and scalability
	DefineScintillatorMaterial();
	DefineCoatingMaterial();
	DefineSiPMMaterial();
}

void DetectorConstruction::DefineScintillatorMaterial()
{
	// I am using a BC-412 scintillator (https://luxiumsolutions.com/radiation-detection-scintillators/plastic-scintillators/bc400-bc404-bc408-bc412-bc416)
	// For simplicity i'll use the NIST material "G4_PLASTIC_SC_VINYLTOLUENE",
	// but i'll leave the code (commented) for the custom material definition (to eventually compare the two)
	#pragma region Scintillator Material Definition
	scint_material = nist->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");

	// Custom material definition (for comparison purposes only)
	// G4Element* H = new G4Element("Hydrogen", "H", 1., 1.00794 * g / mole);
	// G4Element* C = new G4Element("Carbon", "C", 6., 12.0107 * g / mole);
	// 
	// const G4double density = 1.023 * g / cm3;
	// scint_material = new G4Material("BC412_PVT", density, 2);
	// scint_material->AddElement(H, 9);
	// scint_material->AddElement(C, 10);

	#pragma endregion Scintillator Material Definition


	// Optical properties of the scintillator are defined here
	// (For more information on scintillation check https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/TrackingAndPhysics/physicsProcess.html#optical-photon-processes)
	#pragma region Scintillator Optical Properties Definition

	// const G4int nEntries = 3; // for simplicity im using a 3 point spectrum but i may increase it later
	const G4int nEntries = 20; // finally i went for a 20 point spectrum, it should be more than enough
	const G4double sFactor = _scintData.scalingFactor;
	const G4double sYield = _scintData.scintYield;
	const G4double WLpeak = _scintData.waveLengthPeak;	// These values are not used with the current spectrum definition.
	const G4double WL_L = _scintData.waveLengthLeft;	// I plan to implement a function to generate an arbitrary dense spectrum
	const G4double WL_R = _scintData.waveLengthRight;	// using these values in a future revision of the code
	const G4double dTime = _scintData.decayTime;
	const G4double rIndex = _scintData.refractiveIndex;
	const G4double absLength = _scintData.absorptionLength;

	// These will eventually be sampled with arbitrary density from a function using the WLpeak, WL_L and WL_R values
	G4double scintPhotonEnergy[nEntries] = { 2.431*eV, 2.480*eV,2.531*eV,2.583*eV,2.611*eV,2.638*eV,2.667*eV,2.696*eV,2.725*eV,2.756*eV,2.787*eV,2.818*eV,2.851*eV,2.884*eV,2.918*eV,2.952*eV,2.988*eV,3.024*eV,3.062*eV,3.100*eV };
	G4double scintEmission[nEntries] = { 0.0127,0.0154,0.0200,0.0272,0.0318,0.0381,0.0454,0.0544,0.0635,0.0726,0.0817,0.0889,0.0907,0.0889,0.0835,0.0708,0.0526,0.0345,0.0181,0.0091 };
	G4double scintRefractiveIndex[nEntries] = { rIndex , rIndex , rIndex , rIndex , rIndex , rIndex , rIndex, rIndex, rIndex, rIndex, rIndex, rIndex, rIndex, rIndex, rIndex, rIndex, rIndex, rIndex, rIndex, rIndex };
	G4double scintAbsLength[nEntries] = { absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength, absLength };

	// At first i used a 3 point spectrum for simplicity
	// G4double scintPhotonEnergy[nEntries] = { WL_R, WLpeak, WL_L};				// wavelength of the emitted optical photons
	// G4double scintEmission[nEntries] = { 0.1, 1, 0.1 };							// relative intensity of the emitted optical photons
	// G4double scintRefractiveIndex[nEntries] = { rIndex , rIndex , rIndex };		// refractive index of the scintillator (per energy)
	// G4double scintAbsLength[nEntries] = { absLength, absLength, absLength };		// absorption length of the scintillator (per energy)

	G4MaterialPropertiesTable* scintMPT = new G4MaterialPropertiesTable();

	scintMPT->AddProperty("SCINTILLATIONCOMPONENT1", scintPhotonEnergy, scintEmission, nEntries);
	scintMPT->AddProperty("RINDEX", scintPhotonEnergy, scintRefractiveIndex, nEntries);
	scintMPT->AddProperty("ABSLENGTH", scintPhotonEnergy, scintAbsLength, nEntries);

	scintMPT->AddConstProperty("SCINTILLATIONYIELD", sFactor * sYield / MeV);	
	scintMPT->AddConstProperty("SCINTILLATIONYIELD1", 1.0);						
	scintMPT->AddConstProperty("SCINTILLATIONTIMECONSTANT1", dTime);
	scintMPT->AddConstProperty("RESOLUTIONSCALE", 1.0);

	scint_material->SetMaterialPropertiesTable(scintMPT);

	#pragma endregion Scintillator Optical Properties Definition


	// Optical Surface logic of the scintillator is defined here
	// (For more information on boundary surfaces check https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/TrackingAndPhysics/physicsProcess.html#boundary-process)
	#pragma region Scintillator Optical Surface Properties Definition
	
	scint_surface = new G4OpticalSurface("ScintSurface");
	scint_surface->SetType(dielectric_dielectric);
	scint_surface->SetFinish(polished);
	scint_surface->SetModel(unified);
	
	// This is equivalent to 1 - absorption, therefore a value of 1 means no absorption at all

	const G4int rNEntries = 2;
	G4double reflectivity[rNEntries] = { 1., 1. }; // I will probably switch to vectors in a future revision
	
	G4double reflectivityPhotonEnergy[rNEntries] = { 1. * eV, 10. * eV };
	G4MaterialPropertiesTable* scintSurfaceMPT = new G4MaterialPropertiesTable();

	scintSurfaceMPT->AddProperty("REFLECTIVITY", reflectivityPhotonEnergy, reflectivity, rNEntries);
	
	scint_surface->SetMaterialPropertiesTable(scintSurfaceMPT);

	#pragma endregion Scintillator Optical Surface Properties Definition

}

void DetectorConstruction::DefineCoatingMaterial()
{
	// I am using an Aluminum coating, i may consider to change it to TiO2 or Teflon in the future
	
	coating_material = nist->FindOrBuildMaterial("G4_Al");

	// Optical Surface logic of the coating is defined here
	#pragma region Coating Optical Surface Properties Definition
	
	coating_surface = new G4OpticalSurface("CoatingSurface");
	coating_surface->SetType(dielectric_metal);
	coating_surface->SetFinish(polished);

	const G4int nEntries = 2;
	G4double photonEnergy[nEntries] = { 0.1 * eV, 10. * eV};
	
	// I came up with these values, they still need to be verified
	G4double reflectivityCoating[nEntries] = { 0.9, 0.9 };
	G4double specularLobeCoating[nEntries] = { 0.9, 0.9 };
	G4double specularSpikeCoating[nEntries] = { 0., 0. };
	G4double backScatterCoating[nEntries] = { 0., 0. };
	
	G4MaterialPropertiesTable* coatingSurfaceMPT = new G4MaterialPropertiesTable();
	coatingSurfaceMPT->AddProperty("REFLECTIVITY", photonEnergy, reflectivityCoating, nEntries);
	coatingSurfaceMPT->AddProperty("SPECULARLOBECONSTANT", photonEnergy, specularLobeCoating, nEntries);
	coatingSurfaceMPT->AddProperty("SPECULARSPIKECONSTANT", photonEnergy, specularSpikeCoating, nEntries);
	coatingSurfaceMPT->AddProperty("BACKSCATTERCONSTANT", photonEnergy, backScatterCoating, nEntries);

	coating_surface->SetMaterialPropertiesTable(coatingSurfaceMPT);

	#pragma endregion Coating Optical Surface Properties Definition

}

void DetectorConstruction::DefineSiPMMaterial()
{
	// I am using Silicon as SiPM material 
	// (not that it matters since its response won't be based on any physical process)

	sipm_material = nist->FindOrBuildMaterial("G4_Si");

	// Optical properties of the SiPM are defined here
	#pragma region SiPM Optical Properties  Definition

	// The following statement is a strong approximation 
	// since absLenght varies with the wavelength,
	// but it's not really important for the purposes of this project
	const G4double absLength = 100 * nm;

	const G4int nEntries = 2;
	G4double photonEnergy[nEntries] = { 0.1 * eV, 10. * eV };
	G4double siPMRefractiveIndex[nEntries] = { 3.42, 3.42 }; 
	
	G4double siPMAbsLength[nEntries] = { absLength, absLength }; 
	
	G4MaterialPropertiesTable* siPM_MPT = new G4MaterialPropertiesTable();

	siPM_MPT->AddProperty("RINDEX", photonEnergy, siPMRefractiveIndex, nEntries);
	siPM_MPT->AddProperty("ABSLENGTH", photonEnergy, siPMAbsLength, nEntries);

	sipm_material->SetMaterialPropertiesTable(siPM_MPT);

	#pragma endregion SiPM Optical Properties Definition


	// Optical Surface logic of the SiPM is defined here
	// (For more information on boundary surfaces check https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/TrackingAndPhysics/physicsProcess.html#boundary-process)
	#pragma region SiPM Optical Surface Properties Definition

	sipm_surface = new G4OpticalSurface("SiPMSurface");
	sipm_surface->SetType(dielectric_dielectric);
	sipm_surface->SetFinish(polished);
	sipm_surface->SetModel(unified);

	// This is equivalent to 1 - absorption, therefore a value of 1 means no absorption at all
	G4double reflectivity[nEntries] = { 1., 1. }; // I will probably switch to vectors in a future revision

	G4MaterialPropertiesTable* siPMSurfaceMPT = new G4MaterialPropertiesTable();

	siPMSurfaceMPT->AddProperty("REFLECTIVITY", photonEnergy, reflectivity, nEntries);

	scint_surface->SetMaterialPropertiesTable(siPMSurfaceMPT);

	#pragma endregion SiPM Optical Surface Properties Definition

}


// Sensitive Detectors and Fields
// For more information on how these works check https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/Detector/hit.html?highlight=sensitive%20detectors#sensitive-detector
void DetectorConstruction::ConstructSDandField() 
{
	auto* sdManager = G4SDManager::GetSDMpointer();

	// Set the following filter to ignore non-primary muons,
	// be careful that muplus are also ignored.
	auto* muFilter = new PrimaryMuonFilter("PrimaryMuFilter");
	

	#pragma region Scintillator MFD
	
	G4MultiFunctionalDetector* scintMFD = new G4MultiFunctionalDetector("ScintillatorMFD");
	sdManager->AddNewDetector(scintMFD);
	
	// I set these primitive scorers according to the data i want to collect (check the README.md file for that)
	G4PSEnergyDeposit* scintPSedep = new G4PSEnergyDeposit("Edep");
	scintPSedep->SetFilter(muFilter); // to get all particles edep comment out this line
	scintMFD->RegisterPrimitive(scintPSedep);
	
	G4PSPassageTrackLength* scintPSmuPathLength = new G4PSPassageTrackLength("MuPathLength");
	scintPSmuPathLength->SetFilter(muFilter);
	scintMFD->RegisterPrimitive(scintPSmuPathLength);

	SetSensitiveDetector("ScintLogic", scintMFD);
	
	#pragma endregion Scintillator MFD

	
	#pragma region Coating MFD
	
	// It is also possible to add a MFD for each physical copy of the same logical volume
	// but i'm not doing that now since it is not a priority for the purposes of this simulation.
	G4MultiFunctionalDetector* coatingMFD = new G4MultiFunctionalDetector("CoatingMFD");
	sdManager->AddNewDetector(coatingMFD);

	// I set these primitive scorers according to the data i want to collect (check the README.md file for that)
	G4PSEnergyDeposit* coatingPSedep = new G4PSEnergyDeposit("Edep");
	coatingPSedep->SetFilter(muFilter); // to get all particles edep comment out this line
	coatingMFD->RegisterPrimitive(coatingPSedep);

	SetSensitiveDetector("XCoatLogic", coatingMFD);
	SetSensitiveDetector("YCoatLogic", coatingMFD);
	SetSensitiveDetector("FrontCoatLogic", coatingMFD);

	#pragma endregion Coating MFD


	#pragma region SiPM SD & MFD
	
	G4String siliconPMSDName = _siliconPMSDName;
	G4String opCName = _opCName;
	
	SiliconPMSD* siliconPMSD = new SiliconPMSD(siliconPMSDName, opCName);
	sdManager->AddNewDetector(siliconPMSD);
	
	// Assign the SiPMSD to the SiPM logical volume
	SetSensitiveDetector("SiPMLogic", siliconPMSD); // builtin method of G4VUserDetectorConstruction

	#pragma endregion SiPM SD & MFD

	// Test SiPM Bulk MFD
	// G4MultiFunctionalDetector* siPMBulkMFD = new G4MultiFunctionalDetector("SiPMBulkMFD");
	// sdManager->AddNewDetector(siPMBulkMFD);
	// 
	// G4PSEnergyDeposit* siPMBulkPSedep = new G4PSEnergyDeposit("Edep");
	// siPMBulkPSedep->SetFilter(muFilter); // to get all particles edep comment out this line
	// siPMBulkMFD->RegisterPrimitive(siPMBulkPSedep);
	// SetSensitiveDetector("SiPMBulkLogic", siPMBulkMFD);

}