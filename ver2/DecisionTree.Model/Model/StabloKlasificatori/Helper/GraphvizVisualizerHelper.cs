﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace DecisionTree.Model.Model.StabloKlasificatori.Helper;

public static class GraphvizVisualizerHelper
{
    // Ako Graphviz nije u sistemskoj PATH varijabli, ovdje ručno postavite putanju do dot.exe:
    public static string DotExePath = @"C:\Graphviz-12.2.1-win64\bin\dot.exe";

    /// <summary>
    /// Generira .dot datoteku iz stabla klasifikacije i sprema je na zadanu putanju.
    /// Potrebno je imati instaliran Graphviz i postavljen 'dot.exe' u PATH varijabli sustava ili ručno postaviti putanju.
    /// https://graphviz.org/download/ 
    /// exe-instalacija zahtjeva admin permisije ! Nema certifikata. Mnogi korisnici izbjegavaju instalaciju.
    /// zip-verzija ne zahtijeva admin permisije, ali treba ručno postaviti putanju do dot.exe ili je navesti u "DotExePath"
    /// </summary>
    /// <param name="korijen"></param>
    /// <param name="outputFilePath"></param>
    /// <returns></returns>
    public static string MakeDotFile(CvorStabla korijen, string outputFilePath)
    {
        var naziv = Path.GetFileNameWithoutExtension(outputFilePath);
        naziv = Regex.Replace(naziv, @"[^a-zA-Z]", "");
        var sb = new StringBuilder();
        sb.AppendLine($"digraph {naziv} {{");
        sb.AppendLine("node [shape=box];");

        int id = 0;
        Dictionary<CvorStabla, int> nodeIds = new();

        void Print(CvorStabla cvor)
        {
            if (!nodeIds.ContainsKey(cvor))
                nodeIds[cvor] = id++;

            int cvorId = nodeIds[cvor];
            string label = cvor.JeList
                ? $"Klasa: {cvor.Klasa}"
                : cvor.IsNumericki
                    ? $"{cvor.Atribut} <= {cvor.Threshold:0.###}"
                    : $"{cvor.Atribut}";

            sb.AppendLine($"node{cvorId} [label=\"{label}\"];");

            foreach (var dijete in cvor.Djeca)
            {
                var child = dijete.Value;
                if (!nodeIds.ContainsKey(child))
                    nodeIds[child] = id++;

                int childId = nodeIds[child];
                sb.AppendLine($"node{cvorId} -> node{childId} [label=\"{dijete.Key}\"];");

                Print(child);
            }
        }

        Print(korijen);

        sb.AppendLine("}");

        if (!outputFilePath.EndsWith(".dot"))
            outputFilePath = Path.ChangeExtension(outputFilePath, ".dot");

        var folder = Path.GetDirectoryName(outputFilePath);
        if (folder is not null && !Directory.Exists(folder))
        {
            Directory.CreateDirectory(folder);
        }

        File.WriteAllText(outputFilePath, sb.ToString());

        DotFileUSliku(outputFilePath);

        return outputFilePath;
    }


    /// <summary>
    /// Pokreće Graphviz 'dot' alat za konverziju .dot fajla u sliku.
    /// </summary>
    /// <param name="ulazDotFajl">Putanja do .dot fajla (npr. "Files/ime.dot")</param>
    /// <param name="format">Format izlazne slike (npr. "png", "svg")</param>
    public static void DotFileUSliku(string ulazDotFajl, string format = "png")
    {

        var punaPutanjaDot = Path.GetFullPath(ulazDotFajl);
        var punaPutanjaIzlaz = Path.ChangeExtension(punaPutanjaDot, format);

        string arg = $"-T{format} \"{punaPutanjaDot}\" -o \"{punaPutanjaIzlaz}\"";

        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = string.IsNullOrWhiteSpace(DotExePath) ? "dot.exe" : DotExePath,
                Arguments = arg,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = Path.GetDirectoryName(punaPutanjaDot)!
            }
        };

        process.Start();
        string output = process.StandardOutput.ReadToEnd();
        string errors = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode != 0)
        {
            Console.WriteLine($"Graphviz {DotExePath} executable nije pronađen. Provjerite putanju ili instalirajte Graphviz.");
        }
        else
        {
            Console.WriteLine($"Kreiran fajl {punaPutanjaIzlaz}.");
        }
    }
}
