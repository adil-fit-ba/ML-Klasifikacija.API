using DecisionTree.Model.Model.MLP.MLPMreza;

public class MLPKlasifikatorMreza
{
    private List<Layer> Slojevi { get; set; } = new();

    private readonly bool koristiSoftmaxNaIzlazu;

    public MLPKlasifikatorMreza(int brojUlaza, int[] skriveniSlojevi, int brojIzlaza, bool koristiSoftmax = false)
    {
        koristiSoftmaxNaIzlazu = koristiSoftmax;

        int prethodniBroj = brojUlaza;
        foreach (var brojNeurona in skriveniSlojevi)
        {
            Slojevi.Add(new Layer(brojNeurona, prethodniBroj, AktivacijskeFunkcijeHelper.SkriveniSlojevi.ReLU));
            prethodniBroj = brojNeurona;
        }

        if (koristiSoftmax)
        {
            // Zadnji sloj – bez direktne aktivacije za softmax, koristit ćemo sigmoid ili linear
            Slojevi.Add(new Layer(brojIzlaza, prethodniBroj, AktivacijskeFunkcijeHelper.IzlazniSlojevi.Linear));
        }
        else
        {
            // Za binarnu klasifikaciju, koristi Sigmoid direktno
            Slojevi.Add(new Layer(brojIzlaza, prethodniBroj, AktivacijskeFunkcijeHelper.IzlazniSlojevi.Sigmoid));
        }
    }

    public double[] Predikcija(double[] ulazi)
    {
        double[] izlaz = ulazi;
        foreach (var sloj in Slojevi)
        {
            izlaz = sloj.Izracunaj(izlaz);
        }

        // Ako je traženo softmax, primijeni ga na zadnji izlaz
        if (koristiSoftmaxNaIzlazu)
        {
            izlaz = AktivacijskeFunkcijeHelper.IzlazniSlojevi.Softmax(izlaz);
        }

        return izlaz;
    }
}
