using DecisionTree.Model.Model.MLP.Helper;
using DecisionTree.Model.Model.MLP.MLPMreza;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;

namespace DecisionTree.Model.Model.MLP;

/// <summary>
/// MLP klasifikator koji uključuje i logiku same neuronske mreže (slojevi, predikcija).
/// Multy layer perceptron (MLP) je vrsta neuronske mreže koja se sastoji od više slojeva neurona, gdje svaki sloj može imati različit broj neurona.
/// Perceptron je izvedenica od engleske riječi "perceive" (opažati) i "neuron", 
/// 
/// MLP = neuronska mreža s više slojeva
/// https://paarthasaarathi.medium.com/a-complete-guide-to-train-multi-layered-perceptron-neural-networks-3fd8145f9498
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

        // broj epoha treniranja
        public required int BrojEpohaTreniranja = 200;

        // Stopa učenja (learning rate) – određuje veličinu koraka prilikom ažuriranja težina.
        public double UcenjeRate { get; set; } = 0.01;
    }

    public MLPParametri ParametriMLP { get; }

    /// <summary>  
    ///     Konstruktor za inicijalizaciju MLP na osnovu parametara iz json fajla 
    /// </summary>
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

    /// <summary>  
    ///     Konstruktor za inicijalizaciju MLP iz dataset-a. 
    /// </summary>
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
            Slojevi.Add(new Layer(brojNeurona, prethodniBroj, AktivacijskeFunkcijeHelper.SkriveniSlojevi.ReLU, AktivacijskeFunkcijeHelper.SkriveniSlojevi.ReLUDerivacija));
            prethodniBroj = brojNeurona;
        }

        // Izlazni sloj – linearni izlazi ili Sigmoid (softmax se primjenjuje kasnije - u predikciji)
        Slojevi.Add(new Layer(brojIzlaza, prethodniBroj, AktivacijskeFunkcijeHelper.IzlazniSlojevi.Sigmoid, AktivacijskeFunkcijeHelper.IzlazniSlojevi.SigmoidDerivacija));


        for (int epoch = 0; epoch < ParametriMLP.BrojEpohaTreniranja; epoch++)
        {
            double ukupniGubitak = 0.0;
            int brojPrimjera = 0;

            foreach (var red in podaci.Podaci)
            {
                double[] input = MLPDataSetHelper.RedUInputVektor(red.Atributi, this.MLPAtributi);
                double[] ciljniVektor = MLPDataSetHelper.KreirajCiljniVektor(CiljnaKolona, red.Klasa);

                // Trenira i računaj loss
                List<double[]> izlaziPoSlojevima = Trenira(input, ciljniVektor);

                // Loss = MSE (Mean Squared Error)
                double gubitak = 0.0;
                for (int i = 0; i < ciljniVektor.Length; i++)
                    gubitak += Math.Pow(ciljniVektor[i] - izlaziPoSlojevima.Last()[i], 2);

                ukupniGubitak += gubitak;
                brojPrimjera++;
            }

            #region logika_gubitka_i_logovanja
            if (Loguj && (epoch % 10 == 0 || epoch == ParametriMLP.BrojEpohaTreniranja - 1))
            {
                Console.Write($"Epoch: {epoch}, AvgLoss: {(ukupniGubitak / brojPrimjera):F10}  --> ");

                // Loguj izlaz prvog primjera
                var prviRed = podaci.Podaci.First();
                double[] prviInput = MLPDataSetHelper.RedUInputVektor(prviRed.Atributi, this.MLPAtributi);
                double[] prviIzlaz = prviInput;
                foreach (var sloj in Slojevi)
                    prviIzlaz = sloj.Izracunaj(prviIzlaz);

                Console.WriteLine("Primjer izlaza za prvi red: [" + string.Join(", ", prviIzlaz.Select(x => x.ToString("F10"))) + "]");
            }
            #endregion
        }



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

    public List<double[]> Trenira(double[] ulazi, double[] ciljneVrijednosti)
    {
        // FORWARD PASS
        var izlaziPoSlojevima = new List<double[]> { ulazi };
        double[] trenutniUlazi = ulazi;

        foreach (var sloj in Slojevi)
        {
            trenutniUlazi = sloj.Izracunaj(trenutniUlazi);
            izlaziPoSlojevima.Add(trenutniUlazi);
        }

        // BACKWARD PASS
        var izlazniSloj = Slojevi.Last();
        for (int i = 0; i < izlazniSloj.Neuroni.Count; i++)
            izlazniSloj.Neuroni[i].IzracunajDelta(ciljneVrijednosti[i]);

        for (int l = Slojevi.Count - 2; l >= 0; l--)
        {
            var sloj = Slojevi[l];
            var sljedeciSloj = Slojevi[l + 1];

            for (int i = 0; i < sloj.Neuroni.Count; i++)
            {
                double[] tezineSljedecih = sljedeciSloj.Neuroni.Select(n => n.Tezine[i]).ToArray();
                double[] deltaSljedecih = sljedeciSloj.Neuroni.Select(n => n.Delta).ToArray();
                sloj.Neuroni[i].IzracunajDelta(tezineSljedecih, deltaSljedecih);
            }
        }

        // Ažuriranje težina radi logova učenja
        for (int l = 0; l < Slojevi.Count; l++)
        {
            double[] ulaziUSloj = izlaziPoSlojevima[l];
            foreach (var neuron in Slojevi[l].Neuroni)
                neuron.AzurirajTezine(ulaziUSloj, this.ParametriMLP.UcenjeRate);
        }

        return izlaziPoSlojevima;
    }

}
