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
            Neuroni.Add(new Neuron(brojUlazaPoNeuronu, aktivacijskaFunkcija));
    }

    public double[] Izracunaj(double[] ulazi)
    {
        return Neuroni.Select(n => n.Izracunaj(ulazi)).ToArray();
    }
}