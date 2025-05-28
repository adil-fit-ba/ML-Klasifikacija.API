namespace DecisionTree.Model.Model;

public abstract class IKlasifikator
{
    public string Naziv { get; init; }
    public Dictionary<string, object> Parametri { get; init; } = new();
    public Dictionary<string, object> DodatniInfo { get; init; } = new();
    public double VrijemeTreniranjaSek { get; init; }

    protected IKlasifikator(string naziv, object parametri)
    {
        Naziv = naziv;
        Parametri = PretvoriUParametre(parametri);
    }

    private static Dictionary<string, object> PretvoriUParametre(object obj)
    {
        return obj.GetType()
                  .GetProperties()
                  .ToDictionary(
                      prop => prop.Name,
                      prop => prop.GetValue(obj) ?? "null"
                  );
    }

    public abstract string Predikcija(Dictionary<string, VrijednostAtributa> atributi);
}

