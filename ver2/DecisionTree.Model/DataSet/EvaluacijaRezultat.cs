namespace DecisionTree.Model.DataSet;

public class EvaluacijaRezultat
{
    //https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd

    public required string Klasifikator { get; set; }
    public required List<string> Historija { get; set; } = [];
    public required object Parametri { get; set; } = new();    
    public int UkupnoTestiranih { get; set; }
    public int UspjesnoPredvidjeno { get; set; }
    public double Accuracy { get; set; }
    public double AvgPrecision => Precision.Average(x => x.Value);
    public double AvgRecall => Recall.Average(x => x.Value);
    public double AvgF1Score => F1Score.Average(x => x.Value);

    public Dictionary<string, double> Precision { get; set; } = new(); // po klasama
    public Dictionary<string, double> Recall { get; set; } = new(); // po klasama
    public Dictionary<string, double> F1Score { get; set; } = new(); // po klasama
    public Dictionary<string, int> ConfusionMatrix { get; set; } = new();
    public Dictionary<string, int> SveKlase { get; set; } = new(); // broj stvarnih po klasama

    public required List<AtributMeta> AtributiMeta { get; set; } = new();
    public double VrijemeEvaluacijeSek { get; internal set; }
    public double VrijemeTreniranjaSek { get; internal set; }
    public Dictionary<string, object> DodatniInfo { get; internal set; }

    public void Ispisi()
    {
        Console.WriteLine($"Tačnost (Accuracy): {Accuracy:P2}");
        Console.WriteLine();

        var klase = SveKlase.Keys.OrderBy(k => k).ToList();
        foreach (var klasa in klase)
        {
            Console.WriteLine($" 🔹 Klasa: {klasa}");
            Console.WriteLine($"   - Precision: {Precision.GetValueOrDefault(klasa):P2}");
            Console.WriteLine($"   - Recall:    {Recall.GetValueOrDefault(klasa):P2}");
            Console.WriteLine($"   - F1 Score:  {F1Score.GetValueOrDefault(klasa):P2}");
            Console.WriteLine();
        }

        Console.WriteLine("🧾 Confusion Matrix:");
        Console.Write("Stvarna \\ Predikcija\t");
        foreach (var k in klase)
            Console.Write($"{k}\t");
        Console.WriteLine();

        foreach (var stvarna in klase)
        {
            Console.Write($"{stvarna}\t\t");
            foreach (var predikcija in klase)
            {
                var kljuc = $"{stvarna}=>{predikcija}";
                ConfusionMatrix.TryGetValue(kljuc, out int count);
                Console.Write($"{count}\t");
            }
            Console.WriteLine();
        }
    }

}