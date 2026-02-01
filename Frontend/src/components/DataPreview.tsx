import { ArrowLeft, ArrowRight, FileSpreadsheet, Hash, Calendar, Type } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { DatasetInfo } from "@/pages/Index";

interface DataPreviewProps {
  datasetInfo: DatasetInfo;
  onGenerateReport: () => void;
  onBack: () => void;
}

const getTypeIcon = (type: string) => {
  switch (type) {
    case "number":
      return <Hash className="h-3 w-3" />;
    case "date":
      return <Calendar className="h-3 w-3" />;
    default:
      return <Type className="h-3 w-3" />;
  }
};

const getTypeBadgeVariant = (type: string): "default" | "secondary" | "outline" => {
  switch (type) {
    case "number":
      return "default";
    case "date":
      return "secondary";
    default:
      return "outline";
  }
};

export const DataPreview = ({ datasetInfo, onGenerateReport, onBack }: DataPreviewProps) => {
  return (
    <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8">
      {/* Header Section */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-start sm:items-center gap-3 sm:gap-4">
          <div className="flex h-10 w-10 sm:h-12 sm:w-12 shrink-0 items-center justify-center rounded-lg bg-primary/10">
            <FileSpreadsheet className="h-5 w-5 sm:h-6 sm:w-6 text-primary" />
          </div>
          <div className="min-w-0">
            <h1 className="text-xl sm:text-2xl md:text-3xl font-bold truncate">
              {datasetInfo.fileName}
            </h1>
            <p className="text-sm sm:text-base text-muted-foreground">
              {datasetInfo.rows.toLocaleString()} rows • {datasetInfo.columns.length} columns
            </p>
          </div>
        </div>
        
        <div className="flex gap-2 sm:gap-3">
          <Button variant="outline" onClick={onBack} className="gap-2">
            <ArrowLeft className="h-4 w-4" />
            <span className="hidden sm:inline">Back</span>
          </Button>
          <Button onClick={onGenerateReport} className="gap-2 flex-1 sm:flex-none">
            Generate Report
            <ArrowRight className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Column Overview - Mobile Cards / Desktop Grid */}
      <Card>
        <CardHeader className="pb-3 sm:pb-4">
          <CardTitle className="text-lg sm:text-xl">Detected Columns</CardTitle>
          <CardDescription className="text-sm">
            Data types have been automatically identified
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 sm:gap-3">
            {datasetInfo.columns.map((column) => (
              <div
                key={column}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-muted text-sm"
              >
                <span className="font-medium">{column}</span>
                <Badge 
                  variant={getTypeBadgeVariant(datasetInfo.dataTypes[column])}
                  className="text-xs gap-1"
                >
                  {getTypeIcon(datasetInfo.dataTypes[column])}
                  {datasetInfo.dataTypes[column]}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Data Preview Table */}
      <Card>
        <CardHeader className="pb-3 sm:pb-4">
          <CardTitle className="text-lg sm:text-xl">Data Preview</CardTitle>
          <CardDescription className="text-sm">
            Showing first {datasetInfo.preview.length} rows of your dataset
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0 sm:p-6 sm:pt-0">
          {/* Mobile: Stacked Cards View */}
          <div className="block sm:hidden space-y-3 p-4">
            {datasetInfo.preview.map((row, rowIndex) => (
              <div
                key={rowIndex}
                className="p-4 rounded-lg border bg-card space-y-2"
              >
                {datasetInfo.columns.map((column) => (
                  <div key={column} className="flex justify-between items-center text-sm">
                    <span className="text-muted-foreground font-medium">{column}</span>
                    <span className="font-mono">
                      {String(row[column as keyof typeof row] ?? "-")}
                    </span>
                  </div>
                ))}
              </div>
            ))}
          </div>

          {/* Tablet/Desktop: Table View */}
          <div className="hidden sm:block">
            <ScrollArea className="w-full">
              <div className="min-w-[600px]">
                <Table>
                  <TableHeader>
                    <TableRow>
                      {datasetInfo.columns.map((column) => (
                        <TableHead key={column} className="font-semibold whitespace-nowrap">
                          {column}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {datasetInfo.preview.map((row, rowIndex) => (
                      <TableRow key={rowIndex}>
                        {datasetInfo.columns.map((column) => (
                          <TableCell key={column} className="font-mono text-sm whitespace-nowrap">
                            {String(row[column as keyof typeof row] ?? "-")}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </ScrollArea>
          </div>
        </CardContent>
      </Card>

      {/* Data Quality Summary */}
      <Card className="bg-muted/50">
        <CardContent className="p-4 sm:p-6">
          <div className="flex flex-col sm:flex-row sm:items-center gap-4">
            <div className="flex-1">
              <h3 className="font-semibold mb-1">Ready for Analysis</h3>
              <p className="text-sm text-muted-foreground">
                Your dataset looks good! Click "Generate Report" to create your 
                comprehensive analytical report with charts and insights.
              </p>
            </div>
            <Button onClick={onGenerateReport} size="lg" className="w-full sm:w-auto gap-2">
              Generate Report
              <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
