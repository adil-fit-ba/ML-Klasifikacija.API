﻿namespace DecisionTree_1.Controllers;

using DecisionTree.Model.DataSet;
using DecisionTree.Model.Helper;
using DecisionTree.Model.Model;
using DecisionTree.Model.Model.MLP;
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

        fullDataSet.SetCiljnaVarijabla("SalesCategory");
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
                BrojAtributa = null, // null koristi sve atribute
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

        fullDataSet.SetCiljnaVarijabla("SalesCategory");
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
    public IActionResult SalesMLP()
    {
        MLPZahtjev zahtjev = new()
        {
            PutanjaDoFajla = "Files/Sales3.xlsx",
            CiljnaVarijabla = "SalesCategory",
            TestProcenat = 0.5,
            KlasifikatorParamteri = new()
            {
                BrojEpohaTreniranja = 200,
                SkriveniSlojevi = [5, 5]
            }
        };

        MojDataSet fullDataSet = ExcelHelper.Ucitaj(zahtjev.PutanjaDoFajla);

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

        fullDataSet.SetCiljnaVarijabla("SalesCategory");
        fullDataSet.IskljuciAtribute("OutletSales");

        fullDataSet.TransformNumerickuKolonuPoGrupi(
            nazivKolone: "Weight",
            grupnaKolona1: "ProductType",
            grupnaKolona2: "OutletType",
            transformacija: (stara, vrijednostiGrupe) => stara ?? MedianHelper.IzracunajMedijan(vrijednostiGrupe),
            opisTransformacijeZaHistoriju: "medijana"
        );

        fullDataSet.TransformirajKolonuNumericku("Weight", (stara, vrijednostiKolone) => stara ?? MedianHelper.IzracunajMedijan(vrijednostiKolone));
        fullDataSet.StandardizirajSveNumeričkeKolone();
        fullDataSet.NapraviOneHotEncodingSveKolone(ukloniOriginalneKolone: true);

        (MojDataSet treningSet, MojDataSet testSet) = fullDataSet.Podijeli(zahtjev.TestProcenat);

        // Kreiraj i treniraj originalnu mrežu
        MLPKlasifikator mreza = new MLPKlasifikator(treningSet, zahtjev.KlasifikatorParamteri);
        EvaluacijaRezultat rezultatOriginal = fullDataSet.Evaluiraj(mreza, testSet);

        // Snimi mrežu
        string putanjaZaSnimi = "Mreze/Sales3.json";
        var helper = new MLPSerijalizacijaHelper();
        helper.SnimiMrezu(mreza, putanjaZaSnimi);

        // Ponovo ucitaj mrežu
        MLPKlasifikator ucitanaMreza = helper.UcitajMrezu(putanjaZaSnimi);

        // Testiraj ucitanu mrezu
        EvaluacijaRezultat rezultatUcitan = fullDataSet.Evaluiraj(ucitanaMreza, testSet);

        // Vratimo rezultate i provjerimo da li su identični
        bool jeIstaTacnost = Math.Abs(rezultatOriginal.Accuracy - rezultatUcitan.Accuracy) < 1e-6;

        return Ok(new
        {
            originalnaTacnost = rezultatOriginal.Accuracy,
            ucitanaTacnost = rezultatUcitan.Accuracy,
            jeIstaTacnost,
            rezultatOriginal,
            rezultatUcitan
        });
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
    public IActionResult MushroomMLP()
    {
        MLPZahtjev zahtjev = new()
        {
            PutanjaDoFajla = "Files/mushroom1.xlsx",
            CiljnaVarijabla = "class",
            TestProcenat = 0.2,
            KlasifikatorParamteri = new()
            {
                SkriveniSlojevi = [5, 5], //primjer slojeva
                BrojEpohaTreniranja = 10,
            }
        };

        MojDataSet fullDataSet = ExcelHelper.Ucitaj(zahtjev.PutanjaDoFajla, zahtjev.CiljnaVarijabla);
        fullDataSet.NapraviOneHotEncodingSveKolone(ukloniOriginalneKolone: true);

        (MojDataSet treningSet, MojDataSet testSet) = fullDataSet.Podijeli(zahtjev.TestProcenat, random_state: 42);

        MLPKlasifikator stablo = new MLPKlasifikator(treningSet, zahtjev.KlasifikatorParamteri);

        EvaluacijaRezultat rezultat = fullDataSet.Evaluiraj(stablo, testSet);

        return Ok(new
        {
            rezultat,
        });
    }

    [HttpGet]
    public IActionResult BreastCancerWiscoinsinMRandomForest()
    {
        MLPZahtjev zahtjev = new()
        {
            PutanjaDoFajla = "Files/Breast Cancer Wisconsin.xlsx",
            CiljnaVarijabla = "diagnosis",
            TestProcenat = 0.2,
            KlasifikatorParamteri = new()
            {
                SkriveniSlojevi = [5, 5],
                BrojEpohaTreniranja = 500,
            }
        };

        MojDataSet fullDataSet = ExcelHelper.Ucitaj(zahtjev.PutanjaDoFajla);

        fullDataSet.SetCiljnaVarijabla(zahtjev.CiljnaVarijabla);
        fullDataSet.StandardizirajSveNumeričkeKolone();

        (MojDataSet treningSet, MojDataSet testSet) = fullDataSet.Podijeli(zahtjev.TestProcenat, random_state: 42);

        MLPKlasifikator stablo = new MLPKlasifikator(treningSet, zahtjev.KlasifikatorParamteri);

        EvaluacijaRezultat rezultat = fullDataSet.Evaluiraj(stablo, testSet);

        return Ok(new
        {
            rezultat,
        });
    }


    [HttpGet]
    public IActionResult BreastCancerWiscoinsinMLP()
    {
        RandomForstZahtjev zahtjev = new()
        {
            PutanjaDoFajla = "Files/Breast Cancer Wisconsin.xlsx",
            CiljnaVarijabla = "diagnosis",
            TestProcenat = 0.2,
            KlasifikatorParamteri = new()
            {
                BrojAtributa = null, // null koristi sve atribute
                BrojStabala = 10,
                ParametriStabla = new StabloKlasifikator.StabloKlasifikatorParamteri
                {
                    MaxDepth = 5,
                    MinSamples = 5,
                    BrojGrupaZaNumericke = 5
                }
            }
        };


        MojDataSet fullDataSet = ExcelHelper.Ucitaj(zahtjev.PutanjaDoFajla);

        fullDataSet.SetCiljnaVarijabla(zahtjev.CiljnaVarijabla);
        fullDataSet.StandardizirajSveNumeričkeKolone();

        (MojDataSet treningSet, MojDataSet testSet) = fullDataSet.Podijeli(zahtjev.TestProcenat, random_state: 42);

        KlasifikatorBase stablo = new RandomForestKlasifikator(treningSet, zahtjev.KlasifikatorParamteri);

        EvaluacijaRezultat rezultat = fullDataSet.Evaluiraj(stablo, testSet);

        return Ok(new
        {
            rezultat,
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
        public required string PutanjaDoFajla { get; set; } = string.Empty;
        public required string CiljnaVarijabla { get; set; } = string.Empty;
        public required double TestProcenat { get; set; } = 0.2;
        public required StabloKlasifikatorParamteri KlasifikatorParamteri { get; set; } = new();
    }

    public class RandomForstZahtjev
    {
        public required string PutanjaDoFajla { get; set; } = string.Empty;
        public required string CiljnaVarijabla { get; set; } = string.Empty;
        public required double TestProcenat { get; set; } = 0.2;
        public required RandomForestParametri KlasifikatorParamteri { get; set; } = new();
    }

    public class MLPZahtjev
    {
        public required string PutanjaDoFajla { get; set; } = string.Empty;
        public required string CiljnaVarijabla { get; set; } = string.Empty;
        public required double TestProcenat { get; set; } = 0.2;
        public required MLPKlasifikator.MLPParametri KlasifikatorParamteri { get; set; }
    }
}
