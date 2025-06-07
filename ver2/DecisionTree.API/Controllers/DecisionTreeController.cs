namespace DecisionTree_1.Controllers;

using DecisionTree.Model.DataSet;
using DecisionTree.Model.Helper;
using DecisionTree.Model.Model.StabloKlasificatori.Helper;
using Microsoft.AspNetCore.Mvc;
using static RandomForestKlasifikator;
using static StabloKlasifikator;

[ApiController]
[Route("api/[controller]/[action]")]
public class DecisionTreeController : ControllerBase
{

    [HttpGet]
    public IActionResult Play()
    {
        return PokreniTree(new()
        {
            PutanjaDoFajla = "Files/play1.xlsx",
            CiljnaVarijabla = "Play",
            TestProcenat = 0.2,
            KlasifikatorParamteri = new (){
                MaxDepth = 5,
                MinSamples = 5
            }
        });
    }

    [HttpGet]
    public IActionResult Cardio()
    {
        return PokreniTree(new()
        {
            PutanjaDoFajla = "Files/cardio1.xlsx",
            CiljnaVarijabla = "dcRyth",
            TestProcenat = 0.3,
            KlasifikatorParamteri = new()
            {
                MaxDepth = 5,
                MinSamples = 5
            }
        });
    }

    [HttpGet]
    public IActionResult SalesTree()
    {
        StabloZahtjev zahtjev = new()
        {
            PutanjaDoFajla = "Files/Sales3.xlsx",
            CiljnaVarijabla = "SalesCategory",
            TestProcenat = 0.2,
            KlasifikatorParamteri = new()
            {
                MaxDepth = 5,
                MinSamples = 5,
                BrojGrupaZaNumericke = 5
            }
        };

        MojDataSet fullDataSet0 = ExcelHelper.Ucitaj(zahtjev.PutanjaDoFajla);

        MojDataSet fullDataSet = fullDataSet0.Clone();

        var (q1, q3) = KvartilaHelper.IzracunajKvartile(fullDataSet.Podaci, "OutletSales");

        fullDataSet.DodajKolonuKategorijski("SalesCategory", red =>
        {
            if (!red.Atributi.TryGetValue("OutletSales", out var attr) || !attr.Broj.HasValue)
                return null;

            var val = attr.Broj.Value;
            if (val < q1) return "Low";
            if (val > q3) return "High";
            return "Medium";
        });

        fullDataSet.CiljnaKolona = "SalesCategory";
        fullDataSet.IskljuciAtribute("OutletSales");

        fullDataSet.TransformNumerickuKolonuPoGrupi(
            nazivKolone: "Weight",
            grupnaKolona1: "ProductType",
            grupnaKolona2: "OutletType",
            transformacija: (stara, vrijednostiGrupe) => stara ?? MedianHelper.IzracunajMedijan(vrijednostiGrupe),
            opisTransformacijeZaHistoriju: "medijana"
        );

        fullDataSet.TransformirajKolonuNumericku("Weight", (stara, vrijednostiKolone) => stara ?? MedianHelper.IzracunajMedijan(vrijednostiKolone));


        (MojDataSet treningSet, MojDataSet testSet) = fullDataSet.Podijeli(zahtjev.TestProcenat, random_state: 42);

        StabloKlasifikator stablo = new StabloKlasifikator(treningSet, zahtjev.KlasifikatorParamteri);
        GraphvizVisualizerHelper.MakeDotFile(stablo.korijen, "Output/Sales3/Tree");

        EvaluacijaRezultat rezultat = fullDataSet.Evaluiraj(stablo, testSet);

        return Ok(new
        {
            rezultat,
        });
    }

    [HttpGet]
    public IActionResult SalesRandomForest()
    {
        RandomForstZahtjev zahtjev = new()
        {
            PutanjaDoFajla = "Files/Sales3.xlsx",
            CiljnaVarijabla = "SalesCategory",
            TestProcenat = 0.2,
            KlasifikatorParamteri = new()
            {
                BrojAtributa = 4, // koristi sve atribute
                BrojStabala = 10,
                ParametriStabla = new StabloKlasifikator.StabloKlasifikatorParamteri
                {
                    MaxDepth = 5,
                    MinSamples = 5,
                    BrojGrupaZaNumericke = 5
                }
            }
        };

        MojDataSet fullDataSet0 = ExcelHelper.Ucitaj(zahtjev.PutanjaDoFajla);

        MojDataSet fullDataSet = fullDataSet0.Clone();

        var (q1, q3) = KvartilaHelper.IzracunajKvartile(fullDataSet.Podaci, "OutletSales");

        fullDataSet.DodajKolonuKategorijski("SalesCategory", red =>
        {
            if (!red.Atributi.TryGetValue("OutletSales", out var attr) || !attr.Broj.HasValue)
                return null;

            var val = attr.Broj.Value;
            if (val < q1) return "Low";
            if (val > q3) return "High";
            return "Medium";
        });

        fullDataSet.CiljnaKolona = "SalesCategory";
        fullDataSet.IskljuciAtribute("OutletSales");

        fullDataSet.TransformNumerickuKolonuPoGrupi(
            nazivKolone: "Weight",
            grupnaKolona1: "ProductType",
            grupnaKolona2: "OutletType",
            transformacija: (stara, vrijednostiGrupe) => stara ?? MedianHelper.IzracunajMedijan(vrijednostiGrupe),
            opisTransformacijeZaHistoriju: "medijana"
        );

        fullDataSet.TransformirajKolonuNumericku("Weight", (stara, vrijednostiKolone) => stara ?? MedianHelper.IzracunajMedijan(vrijednostiKolone));


        (MojDataSet treningSet, MojDataSet testSet) = fullDataSet.Podijeli(zahtjev.TestProcenat, random_state: 42);

        RandomForestKlasifikator forst = new RandomForestKlasifikator(treningSet, zahtjev.KlasifikatorParamteri);
        for (int i = 0; i < forst.stabla.Count; i++)
        {
            GraphvizVisualizerHelper.MakeDotFile(forst.stabla[i].korijen, $"Output/Sales3/Forest-{i:##}");
        }

        EvaluacijaRezultat rezultat = fullDataSet.Evaluiraj(forst, testSet);

        return Ok(new
        {
            rezultat,
        });
    }


    [HttpGet]
    public IActionResult SalesMultiple()
    {
        var fajl = "Files/Sales3.xlsx";
        var ciljnaKolona = "SalesCategory";
        var testProcenat = 0.2;

        MojDataSet fullDataSet = ExcelHelper.Ucitaj(fajl);

        var (q1, q3) = KvartilaHelper.IzracunajKvartile(fullDataSet.Podaci, "OutletSales");

        fullDataSet.DodajKolonuKategorijski(ciljnaKolona, red =>
        {
            if (!red.Atributi.TryGetValue("OutletSales", out var attr) || !attr.Broj.HasValue)
                return null;

            var val = attr.Broj.Value;
            if (val < q1) return "Low";
            if (val > q3) return "High";
            return "Medium";
        });

        fullDataSet.SetCiljnaVarijabla(ciljnaKolona);
        fullDataSet.IskljuciAtribute("OutletSales");

        fullDataSet.TransformNumerickuKolonuPoGrupi(
            nazivKolone: "Weight",
            grupnaKolona1: "ProductType",
            grupnaKolona2: "OutletType",
            transformacija: (stara, grupe) => stara ?? MedianHelper.IzracunajMedijan(grupe),
            opisTransformacijeZaHistoriju: "medijana"
        );

        fullDataSet.TransformirajKolonuNumericku("Weight", (stara, kolona) => stara ?? MedianHelper.IzracunajMedijan(kolona));

        // za stablo ne treba one-hot encoding
        // fullDataSet.NapraviOneHotEncodingSveKolone();

        // Petlja kroz više konfiguracija stabla
        var rezultati = new List<object>();

        for (int maxDepth = 3; maxDepth <= 10; maxDepth++)
        {
            var parametri = new StabloKlasifikatorParamteri
            {
                MaxDepth = maxDepth,
                MinSamples = 5
            };

            var (treningSet, testSet) = fullDataSet.Podijeli(testProcenat, random_state: 42);
            var stablo = new StabloKlasifikator(treningSet, parametri);
            var rezultat = fullDataSet.Evaluiraj(stablo, testSet);

            rezultati.Add(new
            {
                parametri.MaxDepth,
                rezultat.Accuracy,
                AvgF1 = rezultat.AvgF1Score,
                rezultat.VrijemeTreniranjaSek,
                rezultat.VrijemeEvaluacijeSek
            });
        }

        return Ok(rezultati);
    }

    [HttpGet]
    public IActionResult Mushroom()
    {
        return PokreniTree(new()
        {
            PutanjaDoFajla = "Files/mushroom1.xlsx",
            CiljnaVarijabla = "class",
            TestProcenat = 0.2,
            KlasifikatorParamteri = new()
            {
                MaxDepth = 6,
                MinSamples = 6
            }
        });
    }

    [HttpGet]
    public IActionResult AdultCensusIncome()
    {
        return PokreniTree(new()
        {
            PutanjaDoFajla = "Files/adult-census-income.xlsx",
            CiljnaVarijabla = "class",
            TestProcenat = 0.2,
            KlasifikatorParamteri = new()
            {
                MaxDepth = 8,
                MinSamples = 4
            }
        });
    }
    [HttpGet]
    public IActionResult PokreniTree(
        [FromQuery] StabloZahtjev zahtjev)
    {
        MojDataSet fullDataSet = ExcelHelper.Ucitaj(zahtjev.PutanjaDoFajla, zahtjev.CiljnaVarijabla);
        (MojDataSet treningSet, MojDataSet testSet) = fullDataSet.Podijeli(zahtjev.TestProcenat, random_state: 42);

        StabloKlasifikator stablo = new StabloKlasifikator(treningSet, zahtjev.KlasifikatorParamteri);
        GraphvizVisualizerHelper.MakeDotFile(stablo.korijen, zahtjev.PutanjaDoFajla);
        EvaluacijaRezultat rezultat = fullDataSet.Evaluiraj(stablo, testSet);

        return Ok(new
        {
            rezultat,
        });
    }

    public class StabloZahtjev
    {
        public string PutanjaDoFajla { get; set; } = string.Empty;

        public string CiljnaVarijabla { get; set; } = string.Empty;

        public double TestProcenat { get; set; } = 0.2;

        public StabloKlasifikatorParamteri KlasifikatorParamteri { get; set; } = new();
    }

    public class RandomForstZahtjev
    {
        public string PutanjaDoFajla { get; set; } = string.Empty;

        public string CiljnaVarijabla { get; set; } = string.Empty;

        public double TestProcenat { get; set; } = 0.2;

        public RandomForestParametri KlasifikatorParamteri { get; set; } = new();
    }
}
