using DecisionTree.Model.Model.MLP.Helper;
using DecisionTree.Model.Model.MLP.MLPMreza;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;

namespace DecisionTree.Model.Model.MLP;

/// <summary>
/// MLP klasifikator koji uključuje i logiku same neuronske mreže (slojevi, predikcija).
/// </summary>
public class MLPKlasifikator : KlasifikatorBase
{
    public readonly bool Loguj = true;
    public readonly List<Layer> Slojevi = new();
    public readonly bool koristiSoftmaxNaIzlazu;
    public readonly AtributMeta[] MLPAtributi;
    public AtributMeta CiljnaKolona { get; private set; }

    public class MLPParametri
    {
        // ovo znaci da imamo dva sloja, prvi sa 4 neurona, drugi sa 4 neurona
        public required int[] SkriveniSlojevi { get; set; } = [4, 4];
    }

    public MLPParametri ParametriMLP { get; }
    public MLPKlasifikator(
        MLPParametri parametri,
        bool koristiSoftmaxNaIzlazu,
        AtributMeta ciljanaKolona,
        AtributMeta[] mlpAtributi,
        List<Layer> slojevi)
    : base(nameof(MLPKlasifikator), parametri)
    {
        ParametriMLP = parametri;
        this.koristiSoftmaxNaIzlazu = koristiSoftmaxNaIzlazu;
        CiljnaKolona = ciljanaKolona;
        MLPAtributi = mlpAtributi;
        Slojevi = slojevi;
    }

    public MLPKlasifikator(MojDataSet podaci, MLPParametri parametri)
        : base(nameof(MLPKlasifikator), parametri)
    {
        CiljnaKolona = podaci.CiljnaKolonaMeta;
        ParametriMLP = parametri;

        var stopwatchTreniranje = System.Diagnostics.Stopwatch.StartNew();

        MLPAtributi = podaci.Atributi
            .Where(x => x.KoristiZaModel && x.TipAtributa == TipAtributa.Numericki)
            .ToArray();

        if (CiljnaKolona.TipAtributa == TipAtributa.Numericki)
        {
            throw new InvalidOperationException("MLPKlasifikator podržava samo kategorijske ciljne varijable.");
        }

        int brojIzlaza = CiljnaKolona.Kategoricki!.BrojRazlicitihVrijednosti;
        koristiSoftmaxNaIzlazu = true; // pretpostavljamo višeklasnu klasifikaciju!

        int prethodniBroj = MLPAtributi.Length;
        foreach (var brojNeurona in parametri.SkriveniSlojevi)
        {
            Slojevi.Add(new Layer(brojNeurona, prethodniBroj, AktivacijskeFunkcijeHelper.SkriveniSlojevi.ReLU));
            prethodniBroj = brojNeurona;
        }

        // Izlazni sloj – linearni izlazi ili Sigmoid (softmax se primjenjuje kasnije - u predikciji)
        Slojevi.Add(new Layer(brojIzlaza, prethodniBroj, AktivacijskeFunkcijeHelper.IzlazniSlojevi.Sigmoid));


        stopwatchTreniranje.Stop();
        this.VrijemeTreniranjaSek = stopwatchTreniranje.ElapsedMilliseconds / 1000.0;
    }

    public override string Predikcija(Dictionary<string, VrijednostAtributa> noviCase)
    {
        double[] input = MLPDataSetHelper.RedUInputVektor(noviCase, MLPAtributi);
        double[] izlaz = input;
        foreach (var sloj in Slojevi)
        {
            izlaz = sloj.Izracunaj(izlaz);
        }

        if (koristiSoftmaxNaIzlazu)
        {
            izlaz = AktivacijskeFunkcijeHelper.IzlazniSlojevi.Softmax(izlaz);
        }
        return InterpretirajIzlaz(izlaz);
    }

    private string InterpretirajIzlaz(double[] izlaz)
    {
        int indeksNajvece = Array.IndexOf(izlaz, izlaz.Max());
        var sveKlase = CiljnaKolona.Kategoricki?.SveVrijednosti ?? [];
        return sveKlase.ElementAtOrDefault(indeksNajvece) ?? "Nepoznato";
    }
}
