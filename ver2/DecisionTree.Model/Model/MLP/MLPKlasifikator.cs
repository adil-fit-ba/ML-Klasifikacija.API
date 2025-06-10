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
        public required int[] SkriveniSlojevi { get; set; } = [4, 4];


        public required int BrojEpohaTreniranja = 200;
        /// <summary>
        /// Stopa učenja (learning rate) – određuje veličinu koraka prilikom ažuriranja težina.
        /// </summary>
        public double UcenjeRate { get; set; } = 0.01;
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
                double[] input = MojDataSetHelperMLP.RedUInputVektor(red.Atributi, this.MLPAtributi);
                double[] ciljniVektor = MojDataSetHelperMLP.KreirajCiljniVektor(CiljnaKolona, red.Klasa);

                // Trenira i računaj loss
                var izlaziPoSlojevima = Trenira(input, ciljniVektor);

                // Loss = MSE (Mean Squared Error)
                double gubitak = 0.0;
                for (int i = 0; i < ciljniVektor.Length; i++)
                    gubitak += Math.Pow(ciljniVektor[i] - izlaziPoSlojevima.Last()[i], 2);

                ukupniGubitak += gubitak;
                brojPrimjera++;
            }

            if (Loguj && (epoch % 10 == 0 || epoch == ParametriMLP.BrojEpohaTreniranja - 1))
            {
                Console.Write($"Epoch: {epoch}, AvgLoss: {(ukupniGubitak / brojPrimjera):F10}  --> ");

                // Loguj izlaz prvog primjera
                var prviRed = podaci.Podaci.First();
                double[] prviInput = MojDataSetHelperMLP.RedUInputVektor(prviRed.Atributi, this.MLPAtributi);
                double[] prviIzlaz = prviInput;
                foreach (var sloj in Slojevi)
                    prviIzlaz = sloj.Izracunaj(prviIzlaz);

                Console.WriteLine("Primjer izlaza za prvi red: [" + string.Join(", ", prviIzlaz.Select(x => x.ToString("F10"))) + "]");
            }
        }



        stopwatchTreniranje.Stop();
        this.VrijemeTreniranjaSek = stopwatchTreniranje.ElapsedMilliseconds / 1000.0;
    }

    public override string Predikcija(Dictionary<string, VrijednostAtributa> noviCase)
    {
        double[] input = MojDataSetHelperMLP.RedUInputVektor(noviCase, MLPAtributi);
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

        // Ažuriranje težina
        for (int l = 0; l < Slojevi.Count; l++)
        {
            double[] ulaziUSloj = izlaziPoSlojevima[l];
            foreach (var neuron in Slojevi[l].Neuroni)
                neuron.AzurirajTezine(ulaziUSloj, this.ParametriMLP.UcenjeRate);
        }

        return izlaziPoSlojevima;
    }

}
