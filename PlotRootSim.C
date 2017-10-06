#include <TMap.h>
#include <TTree.h>
#include "TVirtualFFT.h"
#include <TGraph.h>
#include <TCanvas.h>
#include <TCutG.h>
#include <TH2D.h>
#include <THStack.h>
#include <TPaletteAxis.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TH1F.h>
#include <TAxis.h>
#include <TF1.h>
#include <TFile.h>
#include <TImage.h>
#include <TColor.h>
#include "TSpectrum.h"
#include <TVector3.h>
#include <vector>
#include "TPolyMarker.h"

const double degree = 3.14159265359/180.;

TVector3 PoloNord(0.,0.,1.);

#define pi 3.14159265358979323846
#define earthRadiusKm 6371.0

// Display image in canvas and pad.
// 
const double backwall[4]={-30.00,60.03,-171.85,-116.68}; 	// with respect to the Est, in degrees
const double hsite[4] = {1.45,1.45,1.45,1.7}; // altitude asl
const double Rearth = 6371.; // in km
const double SpeedOfLight = 0.3; // in km / usec;
//const double augerCenter[2] = {-35.25, -69.25};
// double longsite[4] = {-69.449673,-69.012203,-69.210845,-69.599633};
 double latsite[4] = {-35.495759,-35.291974,-34.935916,-35.114138};
const int NEYES=4;
const int NTELS=6;
const int NPIXELS=440;
// FD coordinates (in Easting and Northing)
// This function converts decimal degrees to radians
double deg2rad(double deg) {
  return (deg * pi / 180);
}
//  This function converts radians to decimal degrees
double rad2deg(double rad) {
  return (rad * 180 / pi);
}
double longsite[4] = {-69.449673,-69.012203,-69.210845,-69.599633};
//done to stay away from geomertical aberation on lat lon grid
//double latsite[4] = {-0.495759,-0.291974,-0.935916,-0.114138};
const double augerCenter[2] = {-0.25, -69.25};
double Hd=80.; // Height of D layer in km

vector<TCutG*> gcutsList;
vector<Float_t> areaList;
vector<Float_t> distanceList;

TGraph *PixelDirs[NTELS*NEYES];
TGraph *PixelEdges[NPIXELS*NTELS*NEYES];
TGraph *PixelDirsLL[NTELS*NEYES];
TGraph *PixelEdgesLL[NPIXELS*NTELS*NEYES];
double Qtot[NPIXELS];
double Qtmax = 2.e11,Qtmin=-1.e11;
const Int_t NRGBs = 5;
const Int_t MaxColors = 255;
int palette[MaxColors];

double RootOf(double , double , double , int );
void CameraPlot(int , int,int);
void InitPixels();
double distanceEarth(double,double,double,double);
vector<Float_t> v_az(NPIXELS);
vector<Float_t> v_el(NPIXELS);


void PlotRootSim(){
  TTree * tree = new TTree("tree","tree");
  tree->ReadFile("/home/kswiss/Workspace/worktorch/ELVESNet/testData.txt","name/C:lattrue/F:latpred/F:lontrue/F:lonpred/F:isnotblank/I");
  TH2F * histlat = new TH2F("histlat","Latitude MSE Heat Map in Truth Coord. (All Events);True Longitude;True Latitude",16,-68,-60,16,-38,-30);
  TH2F * histlon = new TH2F("histlon","Longitude MSE Heat Map in Truth Coord. (All Events);True Longitude;True Latitude",16,-68,-60,16,-38,-30);
  TH2F * histlatcount = new TH2F("histlatcount","Counts (All Events);True Longitude;True Latitude",16,-68,-60,16,-38,-30);
  TH2F * histloncount = new TH2F("histloncount","Counts (All Events);True Longitude;True Latitude",16,-68,-60,16,-38,-30);
  TH2F * histlatcountzoom = new TH2F("histlatcountzoom","Counts (All Events);True Longitude;True Latitude",16,-68,-60,16,-38,-30);
  TH2F * histloncountzoom = new TH2F("histloncountzoom","Counts (All Events);True Longitude;True Latitude",16,-68,-60,16,-38,-30);
  TH2F * histlatp = new TH2F("histlatp","Latitude MSE Heat Map in Predited Coord. (All Events);Predicted Longitude;Predicted Latitude",16,-68,-60,16,-38,-30);
  TH2F * histlonp = new TH2F("histlonp","Longitude MSE Heat Map in Predicted Coord. (All Events);Predicted Longitude;Predicted Latitude",16,-68,-60,16,-38,-30);
  TH2F * histlatzoom = new TH2F("histlatzoom","Latitude MSE Heat Map in Truth Coord. (w/in train range);True Longitude;True Latitude",16,-68,-63,16,-37,-32);
  TH2F * histlonzoom = new TH2F("histlonzoom","Longitude MSE Heat Map in Truth Coord. (w/in train range);True Longitude;True Latitude",16,-68,-63,16,-37,-32);
  TH2F * histlatpzoom = new TH2F("histlatpzoom","Latitude MSE Heat Map in Predited Coord. (w/in train range);Predicted Longitude;Predicted Latitude",16,-68,-63,16,-37,-32);
  TH2F * histlonpzoom = new TH2F("histlonpzoom","Longitude MSE Heat Map in Predicted Coord. (w/in train range);Predicted Longitude;Predicted Latitude",16,-68,-63,16,-37,-32);
  TH2F * histlatconf = new TH2F("histlatconf","Confusion Matrix;Predicted Latitude;True Latitude",40,-38,-30,40,-38,-30);
  TH2F * histlonconf = new TH2F("histlonconf","Confusion Matrix;Predicted Longitude;True Longitude",40,-68,-60,40,-68,-60);
  TGraph *glatconf = new TGraph(tree->GetEntries());
  TGraph *glonconf = new TGraph(tree->GetEntries());
  TGraph *glatconfOUT = new TGraph(tree->GetEntries());
  TGraph *glonconfOUT = new TGraph(tree->GetEntries());
  float lattrue,lontrue,latpred,lonpred;
  float laterro,lonerro;
  int isnotblank;
  
  tree->SetBranchAddress("lattrue",&lattrue);
  tree->SetBranchAddress("lontrue",&lontrue);
  tree->SetBranchAddress("latpred",&latpred);
  tree->SetBranchAddress("lonpred",&lonpred);
  tree->SetBranchAddress("isnotblank",&isnotblank);

  float counter = 0;
  for(int i = 0; i < tree->GetEntries(); i++){
    tree->GetEntry(i);
    cout << lattrue << " " << latpred << " " << (lattrue-latpred)*(lattrue-latpred) << " || " <<  lontrue << " " << lonpred << " " << (lontrue-lonpred)*(lontrue-lonpred) << endl;
    histlat->Fill(lontrue,lattrue,(lattrue-latpred)*(lattrue-latpred));
    histlon->Fill(lontrue,lattrue,(lontrue-lonpred)*(lontrue-lonpred));
    histlatcount->Fill(lontrue,lattrue);
    histloncount->Fill(lonpred,latpred);
    histlatp->Fill(lonpred,latpred,(lattrue-latpred)*(lattrue-latpred));
    histlonp->Fill(lonpred,latpred,(lontrue-lonpred)*(lontrue-lonpred));
    if(isnotblank){
      histlatzoom->Fill(lontrue,lattrue,(lattrue-latpred)*(lattrue-latpred));
      histlonzoom->Fill(lontrue,lattrue,(lontrue-lonpred)*(lontrue-lonpred));
      histlatcountzoom->Fill(lontrue,lattrue);
      histloncountzoom->Fill(lonpred,latpred);
      histlatpzoom->Fill(lonpred,latpred,(lattrue-latpred)*(lattrue-latpred));
      histlonpzoom->Fill(lonpred,latpred,(lontrue-lonpred)*(lontrue-lonpred));
      laterro += (lattrue-latpred)*(lattrue-latpred);
      lonerro += (lontrue-lonpred)*(lontrue-lonpred);
      //histlatconf->Fill(latpred,lattrue,(lattrue-latpred)*(lattrue-latpred));
      //histlonconf->Fill(lonpred,lontrue,(lontrue-lonpred)*(lontrue-lonpred));
      //histlatconf->Fill(latpred,lattrue);
      //histlonconf->Fill(lonpred,lontrue);
      glatconf->SetPoint(i,latpred,lattrue);
      glonconf->SetPoint(i,lonpred,lontrue);
      counter++;
    }else{
      glatconfOUT->SetPoint(i,latpred,lattrue);
      glonconfOUT->SetPoint(i,lonpred,lontrue);
    }
  }
  for(int i=1; i<=histlat->GetNbinsX(); i++){
    for(int j=1; j<=histlat->GetNbinsY(); j++){
      //if(histlat->GetBinContent(i,j) != 0)   cout<<histlat->GetBinContent(i,j)/ histlatcount->GetBinContent(i,j) << endl;
      if(histlat->GetBinContent(i,j) != 0) histlat->SetBinContent(i,j,histlat->GetBinContent(i,j)/histlatcount->GetBinContent(i,j));
      if(histlatp->GetBinContent(i,j) != 0) histlatp->SetBinContent(i,j,histlatp->GetBinContent(i,j)/histloncount->GetBinContent(i,j));
      if(histlon->GetBinContent(i,j) != 0) histlon->SetBinContent(i,j,histlon->GetBinContent(i,j)/histlatcount->GetBinContent(i,j));
      if(histlonp->GetBinContent(i,j) != 0) histlonp->SetBinContent(i,j,histlonp->GetBinContent(i,j)/histloncount->GetBinContent(i,j));
      if(histlatzoom->GetBinContent(i,j) != 0) histlatzoom->SetBinContent(i,j,histlatzoom->GetBinContent(i,j)/histlatcountzoom->GetBinContent(i,j));
      if(histlatpzoom->GetBinContent(i,j) != 0) histlatpzoom->SetBinContent(i,j,histlatpzoom->GetBinContent(i,j)/histloncountzoom->GetBinContent(i,j));
      if(histlonzoom->GetBinContent(i,j) != 0) histlonzoom->SetBinContent(i,j,histlonzoom->GetBinContent(i,j)/histlatcountzoom->GetBinContent(i,j));
      if(histlonpzoom->GetBinContent(i,j) != 0) histlonpzoom->SetBinContent(i,j,histlonpzoom->GetBinContent(i,j)/histloncountzoom->GetBinContent(i,j));
    }
  }
  laterro = laterro/counter;
  lonerro = lonerro/counter;

  gStyle->SetPalette(54);
  TLine llat(-38,-38,-30,-30);
  TLine llon(-68,-68,-60,-60);

  TCanvas c1("c1","c1",1000,1000);

  gStyle->SetOptStat(0);
  histlat->Draw("colz");
  c1.SaveAs("laterror.png");
  histlon->Draw("colz");
  c1.SaveAs("lonerror.png");
  histlatp->Draw("colz");
  c1.SaveAs("laterrorp.png");
  histlonp->Draw("colz");
  c1.SaveAs("lonerrorp.png");

  histlatzoom->Draw("colz");
  c1.SaveAs("laterrorzoom.png");
  histlonzoom->Draw("colz");
  c1.SaveAs("lonerrorzoom.png");
  histlatpzoom->Draw("colz");
  c1.SaveAs("laterrorpzoom.png");
  histlonpzoom->Draw("colz");
  c1.SaveAs("lonerrorpzoom.png");



  c1.SetLeftMargin(1.8);
  histlatconf->Draw();
  histlatconf->GetYaxis()->SetTitleOffset(1.85);
  histlatconf->SetTitle(TString::Format("Average MSE (w/in range) = %1.2f",laterro));
  histlatconf->GetYaxis()->SetLabelSize(0.026);
  histlatconf->GetXaxis()->SetLabelSize(0.026);
  histlatconf->GetXaxis()->SetTitleSize(0.026);
  histlatconf->GetYaxis()->SetTitleSize(0.026);
  glatconf->SetMarkerSize(1.5);
  glatconf->SetMarkerStyle(20);
  glatconf->SetMarkerColor(kBlue);
  glatconf->Draw("P");
  glatconfOUT->SetMarkerSize(1.5);
  glatconfOUT->SetMarkerStyle(20);
  glatconfOUT->SetMarkerColor(kRed);
  glatconfOUT->Draw("P");
  llat.SetLineWidth(2.0);
  llat.SetLineColor(kBlack);
  llat.Draw("");
  c1.SaveAs("laterrorconf.png");

  c1.SetLeftMargin(1.8);
  histlonconf->Draw();
  histlonconf->GetYaxis()->SetTitleOffset(1.85);
  histlonconf->SetTitle(TString::Format("Average MSE (w/in range) = %1.2f",lonerro));
  histlonconf->GetYaxis()->SetLabelSize(0.026);
  histlonconf->GetXaxis()->SetLabelSize(0.026);
  histlonconf->GetXaxis()->SetTitleSize(0.026);
  histlonconf->GetYaxis()->SetTitleSize(0.026);
  glonconf->SetMarkerSize(1.5);
  glonconf->SetMarkerStyle(20);
  glonconf->SetMarkerColor(kBlue);
  glonconf->Draw("P");
  glonconfOUT->SetMarkerSize(1.5);
  glonconfOUT->SetMarkerStyle(20);
  glonconfOUT->SetMarkerColor(kRed);
  glonconfOUT->Draw("P");
  llon.SetLineWidth(2.0);
  llon.SetLineColor(kBlack);
  llon.Draw("");
  c1.SaveAs("lonerrorconf.png");

  const int NRGBs = 5;
  const int MaxColours = 127;//roberto changed this to 127 in reconstruction
  int palette[MaxColours];
  Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
  Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
  Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
  Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
  Int_t FI = TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, MaxColours);
  for (int i=0;i<MaxColours;i++) {
    palette[i] = FI+i;
    //    cout << palette[i] << endl;
  }
  TColor::SetPalette(MaxColours,palette);
  gStyle->SetNumberContours(MaxColours);//only do the analysis if dealing with the run of choice.

  //fhIntegralPixelPalette[j] = (int)MaxColours * (fhIntegralPixel->GetBinContent(j+1) - 1.0*fhIntegralPixel->GetMinimum())/(1.1*fhIntegralPixel->GetMaximum()- 1.0*fhIntegralPixel->GetMinimum()) + palette[0] ;
  TH1F laterror("laterror","",tree->GetEntries(),1,tree->GetEntries()); 
  TH1F lonerror("lonerror","",tree->GetEntries(),1,tree->GetEntries()); 
  const int nentries = (const int)tree->GetEntries();
  cout << "HERE" << endl;
  int laterrorPalette[nentries];
  int lonerrorPalette[nentries];
  int FOVflag = 1;
  for(int i = 0; i < nentries; i++){
    tree->GetEntry(i);
    if(!FOVflag) isnotblank=1;
    if(isnotblank){
      laterror.Fill(i+1,(lattrue-latpred)*(lattrue-latpred));
      lonerror.Fill(i+1,(lontrue-lonpred)*(lontrue-lonpred));
    }
  }
  const double augerCenter[2] = {-35.25, -69.25};
  TGraph* gAuger = new TGraph(1);
  gAuger->SetPoint(0,augerCenter[1],augerCenter[0]);
  gAuger->SetMarkerStyle(29);
  gAuger->SetMarkerSize(2.5);
  gAuger->SetMarkerColor(kGreen);
  
  TGraph* glaterror[nentries];
  TGraph* glonerror[nentries];
  TGraph* glaterrorp[nentries];
  TGraph* glonerrorp[nentries];
  for(int i = 0; i < nentries; i++){
    tree->GetEntry(i);
    laterrorPalette[i] = (int)MaxColours * (laterror.GetBinContent(i+1) - 1.0*laterror.GetMinimum()) / (1.1*laterror.GetMaximum()- 1.0*laterror.GetMinimum())+ palette[0];
    lonerrorPalette[i] = (int)MaxColours * (lonerror.GetBinContent(i+1) - 1.0*lonerror.GetMinimum()) / (1.1*lonerror.GetMaximum()- 1.0*lonerror.GetMinimum())+ palette[0];
    cout << isnotblank << " " << lattrue << " " << lontrue << " "  << laterrorPalette[i] << " " << lonerrorPalette[i] <<  " " << laterror.GetBinContent(i+1) << " " << lonerror.GetBinContent(i+1) << endl;

    glaterror[i] = new TGraph(1);
    glonerror[i] = new TGraph(1);
    if(!FOVflag) isnotblank=1;
    //    if(isnotblank && laterrorPalette[i]>palette[0]+10){
    if(isnotblank){
      glaterror[i]->SetPoint(i,lontrue,lattrue);
      glaterror[i]->SetMarkerColor(laterrorPalette[i]);
      glaterror[i]->SetMarkerSize(2.0);
      glaterror[i]->SetMarkerStyle(20);
      
      glonerror[i]->SetPoint(i,lontrue,lattrue);
      glonerror[i]->SetMarkerColor(lonerrorPalette[i]);
      glonerror[i]->SetMarkerSize(2.0);
      glonerror[i]->SetMarkerStyle(20);
    }
    glaterrorp[i] = new TGraph(1);
    glonerrorp[i] = new TGraph(1);
	     
    //    if(isnotblank && laterrorPalette[i]>palette[0]+10){
    if(isnotblank){
      glaterrorp[i]->SetPoint(i,lonpred,latpred);
      glaterrorp[i]->SetMarkerColor(laterrorPalette[i]);
      glaterrorp[i]->SetMarkerSize(2.0);
      glaterrorp[i]->SetMarkerStyle(20);
      
      glonerrorp[i]->SetPoint(i,lonpred,latpred);
      glonerrorp[i]->SetMarkerColor(lonerrorPalette[i]);
      glonerrorp[i]->SetMarkerSize(2.0);
      glonerrorp[i]->SetMarkerStyle(20);
    }
  }

  
  TLine l1(-63,-32,-63,-37);
  TLine l2(-68,-32,-68,-37);
  TLine l3(-63,-32,-68,-32);
  TLine l4(-63,-37,-68,-37);
  l1.SetLineColor(kRed);
  l2.SetLineColor(kRed);
  l3.SetLineColor(kRed);
  l4.SetLineColor(kRed);
  TH2F * histlatitude = new TH2F("histlatitude","MSE LATITUDE TRUTH MAP;True Longitude;True Latitude;MSE",40,-70,-58,nentries,-40,-28);
  histlatitude->GetYaxis()->SetTitleOffset(1.85);
  histlatitude->GetYaxis()->SetLabelSize(0.026);
  histlatitude->GetXaxis()->SetLabelSize(0.026);
  histlatitude->GetXaxis()->SetTitleSize(0.026);
  histlatitude->GetYaxis()->SetTitleSize(0.026);
  for (int i=0;i<nentries;i++) {
    histlatitude->SetBinContent(0,i+1,laterror.GetBinContent(i+1));
  }
  histlatitude->GetZaxis()->SetRangeUser(laterror.GetMinimum(),laterror.GetMaximum());
  histlatitude->Draw("colz");
  gPad->Update();
  TPaletteAxis *paletteaxis = (TPaletteAxis*)histlatitude->GetListOfFunctions()->FindObject("palette");
  paletteaxis->SetX1NDC(0.90);
  paletteaxis->SetY1NDC(0.1);
  paletteaxis->SetX2NDC(0.925);
  paletteaxis->SetY2NDC(0.9);
  for(int i = 0; i < nentries; i++){
    glaterror[i]->Draw("P");
  }
  l1.Draw();
  l2.Draw();
  l3.Draw();
  l4.Draw();
  gAuger->Draw("P");
  c1.SaveAs("mselattruth.png");  gAuger->SetMarkerSize(2.5);

  
  histlatitude->SetTitle("MSE LATITUDE PREDICTED MAP;Predicted Longitude;Predicted Latitude;MSE");
  histlatitude->Draw("colz");
  for(int i = 0; i < nentries; i++){
    glaterrorp[i]->Draw("P");
  }
  l1.Draw();
  l2.Draw();
  l3.Draw();
  l4.Draw();
  gAuger->Draw("P");
  c1.SaveAs("mselatpred.png");
  
  histlatitude->SetTitle("MSE LONGITUDE TRUTH MAP;True Longitude;True Latitude;MSE");
  for (int i=0;i<nentries;i++) {
    histlatitude->SetBinContent(0,i+1,lonerror.GetBinContent(i+1));
  }
  histlatitude->GetZaxis()->SetRangeUser(lonerror.GetMinimum(),lonerror.GetMaximum());
  histlatitude->Draw("colz");
  for(int i = 0; i < nentries; i++){
    glonerror[i]->Draw("P");
  }
  l1.Draw();
  l2.Draw();
  l3.Draw();
  l4.Draw();
  gAuger->Draw("P");
  c1.SaveAs("mselontruth.png");
  
  histlatitude->SetTitle("MSE LONGITUDE PREDICTED MAP;Predicted Longitude;Predicted Latitude;MSE");
  histlatitude->Draw("colz");
  for(int i = 0; i < nentries; i++){
    glonerrorp[i]->Draw("P");
  }
  l1.Draw();
  l2.Draw();
  l3.Draw();
  l4.Draw();
  gAuger->Draw("P");
  c1.SaveAs("mselonpred.png");


  TH1F* distanceDev = new TH1F("distanceDev","Prediction Distance from Truth;Distance (km);Number of ELVES",30,0,200);
  TH1F* distanceDevOUT = new TH1F("distanceDev","Prediction Distance from Truth;Distance (km);Number of ELVES",100,0,200);
  for(int i = 0; i < nentries; i++){
    tree->GetEntry(i);
    cout << distanceEarth(lattrue,lontrue,latpred,lonpred) << endl;
    //    distanceDev->Fill(distanceEarth(lattrue,lontrue,latpred,lonpred));
    if(isnotblank) distanceDevOUT->Fill(distanceEarth(lattrue,lontrue,latpred,lonpred));
  }
  //  gStyle->SetOptStat(1);
  TLegend * legend = new TLegend(0.5,0.6,0.9,0.8);
  legend->SetTextSize(0.03);
  legend->SetTextFont(52);
  legend->SetBorderSize(0);
  //legend->AddEntry(distanceDev,"Inside training range","f");
  //legend->AddEntry(distanceDevOUT,"Outside training range","f");
  distanceDevOUT->SetMarkerStyle(20);
  distanceDevOUT->SetMarkerColor(palette[0]+15);
  distanceDevOUT->SetFillColorAlpha(palette[0]+15,0.2);
  distanceDevOUT->SetMarkerSize(1.5);
  distanceDevOUT->Draw("");
  distanceDevOUT->GetYaxis()->SetTitleOffset(1.55);
  distanceDevOUT->GetYaxis()->SetLabelSize(0.026);
  distanceDevOUT->GetXaxis()->SetLabelSize(0.026);
  distanceDevOUT->GetXaxis()->SetTitleSize(0.026);
  distanceDevOUT->GetYaxis()->SetTitleSize(0.026);
  distanceDev->SetMarkerStyle(20);
  distanceDev->SetMarkerColor(palette[0]+45);
  distanceDev->SetFillColorAlpha(palette[0]+45,0.2);
  distanceDev->SetMarkerSize(1.5);
  //distanceDev->Draw("SAME");
  legend->Draw();
  c1.SaveAs("distanceDev.png");
  
}

void CameraPlot(int eye, int mirror, int docut){
  
  //ifstream in;
  //in.open("temp.d");
  gcutsList.clear();
  int site1 = eye-1;
  int tel = mirror-1;
  PixelDirs[site1*NTELS+tel]=new TGraph();
  PixelDirsLL[site1*NTELS+tel]=new TGraph();
  int npix=0;
  for (int nk=0; nk<20 ; nk++){
    for (int nj=0; nj<22 ; nj++){
      int row = nj+1; int col = nk+1;
      double Qt;
      const double eta0 = 0.0261799387;
      const double dOmega = eta0;
      const double dPhi = sqrt(3.) * eta0/2;
      
      const double oo = (col - ((row%2) ? 10. : 10.5)) * dOmega;
      const double ph = (11.66667 - row) * dPhi;
      
      const double mcosOO = -cos(oo);
      const double z = mcosOO * cos(ph);
      const double x = mcosOO * sin(ph);
      const double y = sin(oo);
      TVector3 XPixel(-x,-y,-z);
      XPixel.RotateY(73.5*degree);
      XPixel.RotateZ((backwall[site1]+15.+30.*tel)*degree);
      
      // now we want to plot the marker with Latitude and Longitude
      TVector3 Site1Location(1.,1.,1.);
      Site1Location.SetMag(1.);
      Site1Location.SetPhi(longsite[site1]*degree);
      Site1Location.SetTheta((90.-latsite[site1])*degree);
      double Rsite = Rearth+hsite[site1];
      
      TVector3 East1 = PoloNord.Cross(Site1Location);
      East1.SetMag(1.);
      
      TVector3 AziView1 = East1;
      AziView1.Rotate(XPixel.Phi(),Site1Location);
      TVector3 MaxCirc1 = Site1Location.Cross(AziView1);
      TVector3 PixelHdVert = Site1Location;
      // this is the distance between the FD and the point of light emission, 
      // at Hd altitude (default 90 km)
      double EFD = RootOf(1.,2.*Rsite*cos(XPixel.Theta()),Rsite*Rsite-pow(Rearth+Hd,2),1);
      double EFDTIP = RootOf(1.,2.*Rsite*cos(XPixel.Theta()+0.75*degree),Rsite*Rsite-pow(Rearth+Hd,2),1);
      double PixelEFD = EFD;
      double PixelEFDTIP = EFDTIP;
      double cta = (EFD*EFD-pow(Rearth+Hd,2)-Rsite*Rsite)/(2*Rsite*(Rearth+Hd));
      if (cta>=1 || cta <=-1)cta = 1;
      double alpha_DFD = acos(cta);
      
      // The vertical at FD site is rotated on the max circle to the vertical at light emission
      // and Latitude+Longitude are calculated 
      PixelHdVert.Rotate(-alpha_DFD,MaxCirc1);
      if (PixelHdVert.Angle(Site1Location)>30.*degree) PixelHdVert = - PixelHdVert ; 
      
      double LatPixelHdV = 90.-PixelHdVert.Theta()/degree;
      double LongPixelHdV = PixelHdVert.Phi()/degree;
      
      PixelDirs[site1*NTELS+tel]->SetPoint(npix,XPixel.Phi()/degree,90.-XPixel.Theta()/degree);
      PixelDirsLL[site1*NTELS+tel]->SetPoint(npix,LongPixelHdV,LatPixelHdV);
      double oo2[7], ph2[7];
      oo2[0] = oo+dOmega/2.; ph2[0] = ph-dPhi/3.;
      oo2[1] = oo+dOmega/2.; ph2[1] = ph+dPhi/3.;
      oo2[2] = oo;                     ph2[2] = ph+2.*dPhi/3.;
      oo2[3] = oo-dOmega/2.; ph2[3] = ph+dPhi/3.;
      oo2[4] = oo-dOmega/2.; ph2[4] = ph-dPhi/3.;
      oo2[5] = oo;                     ph2[5] = ph-2.*dPhi/3.;
      oo2[6] = oo2[0];               ph2[6] = ph2[0];
      PixelEdges[npix] = new TGraph();
      PixelEdgesLL[npix] = new TGraph();
      double HexArea = 0;
      
      TString cut_name("gc_pixel_");
      cut_name += nk;
      cut_name += "_";
      cut_name += nj;
      TCutG* gc_pixel = new TCutG(cut_name,7);
      if(docut){
	gc_pixel->SetVarX("longitude");
	gc_pixel->SetVarY("latitude");
      }
      for (int iV = 0; iV<7 ; iV++){
	double mcosOO2 = -cos(oo2[iV]);
	double z2 = mcosOO2 * cos(ph2[iV]);
	double x2 = mcosOO2 * sin(ph2[iV]);
	double y2 = sin(oo2[iV]);
	
	TVector3 *XPixEdge = new TVector3(-x2,-y2,-z2);
	XPixEdge->RotateY(73.5*degree);
	XPixEdge->RotateZ((backwall[site1]+15.+30.*tel)*degree);
	
	PixelEdges[npix]->SetPoint(iV,XPixEdge->Phi()/degree,90.-XPixEdge->Theta()/degree);
	TVector3 AziView2 = East1;
	AziView2.Rotate(XPixEdge->Phi(),Site1Location);
	TVector3 MaxCirc2 = Site1Location.Cross(AziView2);
	TVector3 PixEdgeHdVert = Site1Location;
	// this is the distance between the FD and the point of light emission, 
	// at Hd altitude (default 90 km)
	EFD = RootOf(1.,2.*Rsite*cos(XPixEdge->Theta()),Rsite*Rsite-pow(Rearth+Hd,2),1);
	cta = (EFD*EFD-pow(Rearth+Hd,2)-Rsite*Rsite)/(2*Rsite*(Rearth+Hd));
	if (cta>=1 || cta <=-1)cta = 1;
	alpha_DFD = acos(cta);
	
	PixEdgeHdVert.Rotate(-alpha_DFD,MaxCirc2);
	if (PixEdgeHdVert.Angle(Site1Location)>30.*degree) PixEdgeHdVert = - PixEdgeHdVert ; 
	
	double LatPixEdgeHdV = 90.-PixEdgeHdVert.Theta()/degree;
	double LongPixEdgeHdV = PixEdgeHdVert.Phi()/degree;
	PixelEdgesLL[npix]->SetPoint(iV,LongPixEdgeHdV,LatPixEdgeHdV);

	if(docut)  gc_pixel->SetPoint(iV,LongPixEdgeHdV,LatPixEdgeHdV);	

	if (iV>0) {
	  Double_t xLong, xLat;  PixelEdgesLL[npix]->GetPoint(iV-1,xLong,xLat);
	  double xth = (90.-xLat)*degree;
	  double xph = xLong*degree;
	  
	  TVector3 xPixEdge(sin(xth)*cos(xph),sin(xth)*sin(xph),cos(xth));
	  double L0 = (Rearth+Hd)*xPixEdge.Angle(PixEdgeHdVert);
	  double L1 = (Rearth+Hd)*PixelHdVert.Angle(PixEdgeHdVert);
	  double L2 = (Rearth+Hd)*PixelHdVert.Angle(xPixEdge);
	  double SP = (L0+L1+L2)/2.;
	  double Area = sqrt(SP*(SP-L0)*(SP-L1)*(SP-L2));
	  HexArea += Area;
	}
      }
      if(docut){
	gcutsList.push_back(gc_pixel);
	// gc_pixel->SetFillColorAlpha(kRed+1,0.3);
	// gc_pixel->Draw("f");
	areaList.push_back(HexArea);
	distanceList.push_back(PixelEFD);
	//	cout << PixelEFDTIP << endl;
      }

      PixelEdgesLL[npix]->SetLineColor(mirror-1);
      if (row == 2) PixelEdgesLL[npix]->SetLineColor(1);
      if (col == 2) PixelEdgesLL[npix]->SetLineColor(1);
      int qT = (int) (MaxColors*(HexArea)/(3000.));
      PixelEdgesLL[npix]->Draw("L");
      npix++;
      
    }
  }
  
  
  PixelDirsLL[site1*NTELS+tel]->Draw("P");
}
double RootOf(double a, double b, double c, int i)
{
  
  double d = b*b-4*a*c;
  double r = 9.999e99;
  if (d >= 0){ 
    r = (-b-sqrt(d))/(2*a) ;
    if (i == 1) r = (-b+sqrt(d))/(2*a);
  }
  return r;
}
void InitPixels() 
{
    Float_t az[NPIXELS] = {13.51, 14.27, 13.54, 14.33, 13.61, 14.41, 13.70, 14.52, 13.82, 14.66,
      13.97, 14.83, 14.15, 15.04, 14.36, 15.28, 14.61, 15.56, 14.90, 15.89,
      15.23, 16.26, 12.01, 12.77, 12.04, 12.82, 12.10, 12.89, 12.18, 12.99,
      12.28, 13.12, 12.42, 13.27, 12.58, 13.46, 12.77, 13.68, 13.00, 13.94,
      13.25, 14.23, 13.55, 14.56, 10.51, 11.27, 10.54, 11.31, 10.58, 11.38,
      10.66, 11.46, 10.75, 11.58, 10.87, 11.72, 11.01, 11.88, 11.18, 12.08,
      11.38, 12.30, 11.61, 12.57, 11.87, 12.86,  9.01,  9.77,  9.03,  9.80,
       9.07,  9.86,  9.13,  9.94,  9.22, 10.04,  9.32, 10.16,  9.44, 10.30,
       9.59, 10.47,  9.76, 10.67,  9.95, 10.90, 10.18, 11.16,  7.51,  8.26,
       7.53,  8.30,  7.56,  8.34,  7.61,  8.41,  7.68,  8.49,  7.77,  8.60,
       7.87,  8.72,  7.99,  8.87,  8.14,  9.03,  8.30,  9.23,  8.49,  9.45,
       6.00,  6.76,  6.02,  6.79,  6.05,  6.83,  6.09,  6.88,  6.15,  6.95,
       6.21,  7.03,  6.30,  7.14,  6.40,  7.26,  6.51,  7.40,  6.64,  7.56,
       6.80,  7.74,  4.50,  5.26,  4.52,  5.28,  4.54,  5.31,  4.57,  5.35,
       4.61,  5.41,  4.66,  5.47,  4.72,  5.55,  4.80,  5.65,  4.88,  5.75,
       4.98,  5.88,  5.10,  6.02,  3.00,  3.76,  3.01,  3.77,  3.02,  3.79,
       3.05,  3.82,  3.07,  3.86,  3.11,  3.91,  3.15,  3.97,  3.20,  4.03,
       3.26,  4.11,  3.32,  4.20,  3.40,  4.30,  1.50,  2.25,  1.51,  2.26,
       1.51,  2.28,  1.52,  2.29,  1.54,  2.32,  1.55,  2.35,  1.57,  2.38,
       1.60,  2.42,  1.63,  2.47,  1.66,  2.52,  1.70,  2.58,  0.00,  0.75,
       0.00,  0.75,  0.00,  0.76,  0.00,  0.76,  0.00,  0.77,  0.00,  0.78,
       0.00,  0.79,  0.00,  0.81,  0.00,  0.82,  0.00,  0.84,  0.00,  0.86,
      -1.50, -0.75, -1.51, -0.75, -1.51, -0.76, -1.52, -0.76, -1.54, -0.77,
      -1.55, -0.78, -1.57, -0.79, -1.60, -0.81, -1.63, -0.82, -1.66, -0.84,
      -1.70, -0.86, -3.00, -2.25, -3.01, -2.26, -3.02, -2.28, -3.05, -2.29,
      -3.07, -2.32, -3.11, -2.35, -3.15, -2.38, -3.20, -2.42, -3.26, -2.47,
      -3.32, -2.52, -3.40, -2.58, -4.50, -3.76, -4.52, -3.77, -4.54, -3.79,
      -4.57, -3.82, -4.61, -3.86, -4.66, -3.91, -4.72, -3.97, -4.80, -4.03,
      -4.88, -4.11, -4.98, -4.20, -5.10, -4.30, -6.00, -5.26, -6.02, -5.28,
      -6.05, -5.31, -6.09, -5.35, -6.15, -5.41, -6.21, -5.47, -6.30, -5.55,
      -6.40, -5.65, -6.51, -5.75, -6.64, -5.88, -6.80, -6.02, -7.51, -6.76,
      -7.53, -6.79, -7.56, -6.83, -7.61, -6.88, -7.68, -6.95, -7.77, -7.03,
      -7.87, -7.14, -7.99, -7.26, -8.14, -7.40, -8.30, -7.56, -8.49, -7.74,
      -9.01, -8.26, -9.03, -8.30, -9.07, -8.34, -9.13, -8.41, -9.22, -8.49,
      -9.32, -8.60, -9.44, -8.72, -9.59, -8.87, -9.76, -9.03, -9.95, -9.23,
     -10.18, -9.45,-10.51, -9.77,-10.54, -9.80,-10.58, -9.86,-10.66, -9.94,
     -10.75,-10.04,-10.87,-10.16,-11.01,-10.30,-11.18,-10.47,-11.38,-10.67,
     -11.61,-10.90,-11.87,-11.16,-12.01,-11.27,-12.04,-11.31,-12.10,-11.38,
     -12.18,-11.46,-12.28,-11.58,-12.42,-11.72,-12.58,-11.88,-12.77,-12.08,
     -13.00,-12.30,-13.25,-12.57,-13.55,-12.86,-13.51,-12.77,-13.54,-12.82,
     -13.61,-12.89,-13.70,-12.99,-13.82,-13.12,-13.97,-13.27,-14.15,-13.46,
     -14.36,-13.68,-14.61,-13.94,-14.90,-14.23,-15.23,-14.56,-15.01,-14.27,
     -15.05,-14.33,-15.12,-14.41,-15.22,-14.52,-15.35,-14.66,-15.51,-14.83,
     -15.71,-15.04,-15.95,-15.28,-16.22,-15.56,-16.54,-15.89,-16.90,-16.26};

 Float_t el[NPIXELS] = { 2.08,  3.34,  4.61,  5.85,  7.14,  8.37,  9.66, 10.89, 12.18, 13.40,
      14.71, 15.91, 17.23, 18.42, 19.74, 20.93, 22.26, 23.44, 24.77, 25.94,
      27.28, 28.43,  2.10,  3.36,  4.64,  5.89,  7.18,  8.42,  9.72, 10.96,
      12.26, 13.49, 14.80, 16.02, 17.33, 18.54, 19.87, 21.07, 22.40, 23.59,
      24.93, 26.11, 27.46, 28.63,  2.11,  3.38,  4.66,  5.92,  7.22,  8.47,
       9.77, 11.02, 12.32, 13.56, 14.87, 16.11, 17.43, 18.65, 19.97, 21.19,
      22.52, 23.73, 25.07, 26.27, 27.61, 28.80,  2.12,  3.39,  4.68,  5.95,
       7.25,  8.51,  9.81, 11.07, 12.38, 13.63, 14.94, 16.19, 17.51, 18.75,
      20.07, 21.30, 22.63, 23.86, 25.19, 26.41, 27.75, 28.96,  2.13,  3.41,
       4.70,  5.98,  7.28,  8.55,  9.85, 11.12, 12.43, 13.69, 15.00, 16.26,
      17.58, 18.83, 20.15, 21.39, 22.72, 23.96, 25.29, 26.53, 27.86, 29.09,
       2.13,  3.42,  4.72,  6.00,  7.30,  8.58,  9.88, 11.16, 12.47, 13.74,
      15.05, 16.32, 17.63, 18.89, 20.21, 21.47, 22.80, 24.05, 25.38, 26.62,
      27.96, 29.20,  2.14,  3.43,  4.73,  6.02,  7.32,  8.60,  9.91, 11.19,
      12.50, 13.78, 15.09, 16.36, 17.68, 18.95, 20.26, 21.53, 22.85, 24.12,
      25.44, 26.70, 28.03, 29.29,  2.14,  3.44,  4.74,  6.03,  7.33,  8.62,
       9.92, 11.21, 12.52, 13.80, 15.11, 16.40, 17.71, 18.99, 20.30, 21.58,
      22.89, 24.17, 25.49, 26.76, 28.08, 29.35,  2.14,  3.44,  4.74,  6.04,
       7.34,  8.63,  9.93, 11.23, 12.53, 13.82, 15.13, 16.42, 17.73, 19.02,
      20.32, 21.61, 22.92, 24.21, 25.52, 26.80, 28.11, 29.40,  2.14,  3.44,
       4.74,  6.04,  7.34,  8.64,  9.94, 11.24, 12.54, 13.83, 15.13, 16.43,
      17.73, 19.03, 20.33, 21.63, 22.93, 24.22, 25.53, 26.82, 28.12, 29.42,
       2.14,  3.44,  4.74,  6.04,  7.34,  8.64,  9.93, 11.24, 12.53, 13.83,
      15.13, 16.43, 17.73, 19.03, 20.32, 21.63, 22.92, 24.22, 25.52, 26.82,
      28.11, 29.42,  2.14,  3.44,  4.74,  6.04,  7.33,  8.63,  9.92, 11.23,
      12.52, 13.82, 15.11, 16.42, 17.71, 19.02, 20.30, 21.61, 22.89, 24.21,
      25.49, 26.80, 28.08, 29.40,  2.14,  3.44,  4.73,  6.03,  7.32,  8.62,
       9.91, 11.21, 12.50, 13.80, 15.09, 16.40, 17.68, 18.99, 20.26, 21.58,
      22.85, 24.17, 25.44, 26.76, 28.03, 29.35,  2.13,  3.43,  4.72,  6.02,
       7.30,  8.60,  9.88, 11.19, 12.47, 13.78, 15.05, 16.36, 17.63, 18.95,
      20.21, 21.53, 22.80, 24.12, 25.38, 26.70, 27.96, 29.29,  2.13,  3.42,
       4.70,  6.00,  7.28,  8.58,  9.85, 11.16, 12.43, 13.74, 15.00, 16.32,
      17.58, 18.89, 20.15, 21.47, 22.72, 24.05, 25.29, 26.62, 27.86, 29.20,
       2.12,  3.41,  4.68,  5.98,  7.25,  8.55,  9.81, 11.12, 12.38, 13.69,
      14.94, 16.26, 17.51, 18.83, 20.07, 21.39, 22.63, 23.96, 25.19, 26.53,
      27.75, 29.09,  2.11,  3.39,  4.66,  5.95,  7.22,  8.51,  9.77, 11.07,
      12.32, 13.63, 14.87, 16.19, 17.43, 18.75, 19.97, 21.30, 22.52, 23.86,
      25.07, 26.41, 27.61, 28.96,  2.10,  3.38,  4.64,  5.92,  7.18,  8.47,
       9.72, 11.02, 12.26, 13.56, 14.80, 16.11, 17.33, 18.65, 19.87, 21.19,
      22.40, 23.73, 24.93, 26.27, 27.46, 28.80,  2.08,  3.36,  4.61,  5.89,
       7.14,  8.42,  9.66, 10.96, 12.18, 13.49, 14.71, 16.02, 17.23, 18.54,
      19.74, 21.07, 22.26, 23.59, 24.77, 26.11, 27.28, 28.63,  2.07,  3.34,
       4.58,  5.85,  7.09,  8.37,  9.60, 10.89, 12.10, 13.40, 14.61, 15.91,
      17.11, 18.42, 19.61, 20.93, 22.10, 23.44, 24.60, 25.94, 27.09, 28.43};

 for(Int_t i = 0; i<NPIXELS; i++){
   v_el[i] = el[i]; v_az[i] = az[i];
 }

}

double distanceEarth(double lat1d, double lon1d, double lat2d, double lon2d) {
  double lat1r, lon1r, lat2r, lon2r, u, v;
  lat1r = deg2rad(lat1d);
  lon1r = deg2rad(lon1d);
  lat2r = deg2rad(lat2d);
  lon2r = deg2rad(lon2d);
  u = sin((lat2r - lat1r)/2);
  v = sin((lon2r - lon1r)/2);
  return 2.0 * earthRadiusKm * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
}
