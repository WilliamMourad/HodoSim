#pragma once

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"

#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Element.hh"

#include "G4OpticalSurface.hh"
#include "G4LogicalBorderSurface.hh"


struct ReferenceFrame {
	// -- Still Not Implemented --
	// Set the position and rotation of the detector reference frame respect to the world frame
	// This is useful to place the scintillator and the SiPM in the correct position
	G4ThreeVector position;
	G4RotationMatrix* rotation;
};

struct BoxGeometry {
	G4double sizeX;
	G4double sizeY;
	G4double sizeZ;
};

struct ScintillatorProperties {
	BoxGeometry geometry;

	G4double scalingFactor;			// scaling on scintillation yield
	G4double scintYield;			// how many photons are emitted per MeV
	G4double waveLengthPeak;		// peak energy of the emitted photons (in eV)
	G4double waveLengthLeft;		// left limit of the energy of the emitted photons (in eV)
	G4double waveLengthRight;		// right limit of the energy of the emitted photons (in eV)
	G4double decayTime;				// time constant for the decay of the scintillation light (in ns)
	G4double refractiveIndex;		// 
	G4double absorptionLength;		// 

};

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
	DetectorConstruction(
		G4double worldSizeXYZ, 
		ScintillatorProperties scintData, 
		G4double coatingThickness,
		G4double siPMThickness,
		G4double gap,
		G4String siliconPMSDName,
		G4String scintLVName,
		G4String opCName,
		G4bool enableCuts
	);
	~DetectorConstruction();

	G4VPhysicalVolume* Construct();
	void ConstructSDandField();

private:
	G4VPhysicalVolume* BuildGeometry();
	void DefineMaterials();
	void DefineScintillatorMaterial();
	void DefineCoatingMaterial();
	void DefineSiPMMaterial();
	
	G4double _worldSizeXYZ;
	ScintillatorProperties _scintData;
	G4double _coatingThickness;
	G4double _siPMThickness;
	G4double _gap;

	G4String _scintLVName;
	G4String _siliconPMSDName;
	G4String _opCName; // The collection name used for OpticalPhotonHit collection

	G4bool _enableCuts;

	G4NistManager* nist;

	G4Material* vacuum;
	G4Material* air;
	G4Material* scint_material;
	G4Material* coating_material;
	G4Material* sipm_material;

	G4OpticalSurface* coating_surface;
	G4OpticalSurface* scint_surface;
	G4OpticalSurface* sipm_surface;

	G4LogicalBorderSurface* scint_to_coating_top;
	G4LogicalBorderSurface* scint_to_coating_bottom;
	G4LogicalBorderSurface* scint_to_coating_left;
	G4LogicalBorderSurface* scint_to_coating_right;
	G4LogicalBorderSurface* scint_to_coating_front;
	G4LogicalBorderSurface* coating_top_to_scint;
	G4LogicalBorderSurface* coating_bottom_to_scint;
	G4LogicalBorderSurface* coating_left_to_scint;
	G4LogicalBorderSurface* coating_right_to_scint;
	G4LogicalBorderSurface* coating_front_to_scint;
	G4LogicalBorderSurface* scint_to_sipm;
	G4LogicalBorderSurface* sipm_to_scint;

	// For a better implementation I should consider adding also some class variables
	// for the names/instances of volumes/borders, initializing the names in the constructor.
	// (in order to easily use them within other classes)
};
