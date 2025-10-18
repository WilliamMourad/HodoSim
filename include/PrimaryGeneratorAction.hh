#pragma once

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4GeneralParticleSource.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"


struct ParticleGunSettings {
    G4bool isActive;
    G4int particleN;
    G4double particleEnergy;
    G4ThreeVector gunPosition;
    G4ThreeVector gunDirection;
};

struct GPSSettings {
    G4bool isActive;
	G4int particleN;
	G4double particleMeanEnergy;
	G4double particleEnergySigma;
	G4ThreeVector gpsPosition;
    G4double gpsRadius;
	G4ThreeVector gpsRot1;
	G4ThreeVector gpsRot2;
	G4double beamApertureX;
	G4double beamApertureY;
};

struct PrimaryGeneratorActionParameters {
    G4String particleName;
    ParticleGunSettings particleGunSettings;
    GPSSettings gpsSettings;
};

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
    PrimaryGeneratorAction(PrimaryGeneratorActionParameters primaryGeneratorActionParameters);
    ~PrimaryGeneratorAction();

    void GeneratePrimaries(G4Event* anEvent);

private:

    void BuildParticleGun();
    void BuildGPS();
    
	PrimaryGeneratorActionParameters _primaryGeneratorActionParameters;
    G4ParticleGun* particleGun;
    G4GeneralParticleSource* gps;
};
