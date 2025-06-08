using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree.Model.Model.MLP.MLPMreza;

public class Layer
{
    public List<Neuron> Neuroni { get; set; } = new();

    public Layer(int brojNeurona, int brojUlazaPoNeuronu, Func<double, double> aktivacijskaFunkcija)
    {
        for (int i = 0; i < brojNeurona; i++)
        {
            Neuroni.Add(new Neuron(brojUlazaPoNeuronu, aktivacijskaFunkcija));
        }
    }

    // Računa izlaz cijelog sloja
    public double[] Izracunaj(double[] ulazi)
    {
        var izlazi = new double[Neuroni.Count];
        for (int i = 0; i < Neuroni.Count; i++)
        {
            izlazi[i] = Neuroni[i].Izracunaj(ulazi);
        }
        return izlazi;
    }
}
